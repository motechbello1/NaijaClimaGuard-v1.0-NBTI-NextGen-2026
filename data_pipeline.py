"""
NaijaClimaGuard — Data Pipeline (data_pipeline.py)
====================================================
Phase 1: Data Engine for TRL-5 Flood Prediction Model

Sources:
  - Open-Meteo Historical Weather API (zero key required)
  - Open-Meteo Flood API / GloFAS reanalysis (zero key required)

Output:
  - training_data.csv  : cleaned, labelled dataset ready for ML training
  - pipeline_audit.log : full execution trace for TRL documentation
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = Path("pipeline_audit.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("naija_clima_guard.pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LocationConfig:
    name: str
    state: str
    latitude: float
    longitude: float
    flood_site: bool
    flood_year: Optional[int] = None


@dataclass
class PipelineConfig:
    historical_start: str = "2018-01-01"
    historical_end: str = "2023-06-30"

    flood_discharge_percentile: float = 90.0
    flood_rainfall_threshold_mm: float = 50.0

    # FIXED: only valid daily variables for Open-Meteo archive endpoint.
    # soil_moisture_0_to_7cm is hourly-only — removed.
    # surface_pressure is hourly-only — removed.
    # Replaced with precipitation_hours (wetness proxy) and wind_speed_10m_max.
    weather_api: str = "https://archive-api.open-meteo.com/v1/archive"
    flood_api: str = "https://flood-api.open-meteo.com/v1/flood"

    output_csv: Path = Path("training_data.csv")

    max_retries: int = 3
    backoff_factor: float = 2.0
    request_timeout: int = 60

    locations: list = field(default_factory=lambda: [
        LocationConfig(
            name="Lokoja", state="Kogi",
            latitude=7.7975, longitude=6.7399,
            flood_site=True, flood_year=2022,
        ),
        LocationConfig(
            name="Makurdi", state="Benue",
            latitude=7.7337, longitude=8.5227,
            flood_site=True, flood_year=2022,
        ),
        LocationConfig(
            name="Onitsha", state="Anambra",
            latitude=6.1667, longitude=6.7833,
            flood_site=True, flood_year=2022,
        ),
        LocationConfig(
            name="Kano", state="Kano",
            latitude=12.0022, longitude=8.5920,
            flood_site=False,
        ),
        LocationConfig(
            name="Maiduguri", state="Borno",
            latitude=11.8333, longitude=13.1500,
            flood_site=False,
        ),
    ])


# ---------------------------------------------------------------------------
# HTTP Session
# ---------------------------------------------------------------------------
class ResilientSession:
    def __init__(self, config: PipelineConfig):
        self.timeout = config.request_timeout
        session = requests.Session()
        retry_strategy = Retry(
            total=config.max_retries,
            backoff_factor=config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self.session = session

    def get(self, url: str, params: dict) -> dict:
        response = self.session.get(url, params=params, timeout=self.timeout)
        if not response.ok:
            logger.error("API error %d: %s", response.status_code, response.text[:400])
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Weather Fetcher
# ---------------------------------------------------------------------------
class WeatherFetcher:
    """
    Valid Open-Meteo daily archive variables used here:
      precipitation_sum          — daily rainfall total (mm)
      precipitation_hours        — hours of rain per day (saturation proxy)
      temperature_2m_max         — max temp (°C)
      temperature_2m_min         — min temp (°C)
      wind_speed_10m_max         — max wind speed (km/h), storm proxy
      et0_fao_evapotranspiration — reference ET (mm), drainage proxy

    REMOVED (not available as daily):
      soil_moisture_0_to_7cm  -> hourly only
      surface_pressure        -> hourly only
    """

    VARIABLES = [
        "precipitation_sum",
        "precipitation_hours",
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_max",
        "et0_fao_evapotranspiration",
    ]

    def __init__(self, config: PipelineConfig, session: ResilientSession):
        self.config = config
        self.session = session

    def fetch(self, loc: LocationConfig) -> pd.DataFrame:
        logger.info("[WEATHER] Fetching for %s, %s ...", loc.name, loc.state)
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "start_date": self.config.historical_start,
            "end_date": self.config.historical_end,
            "daily": ",".join(self.VARIABLES),
            "timezone": "Africa/Lagos",
        }
        try:
            data = self.session.get(self.config.weather_api, params)
        except requests.exceptions.RequestException as exc:
            logger.error("[WEATHER] Request failed for %s: %s", loc.name, exc)
            raise

        daily = data.get("daily", {})
        if not daily or "time" not in daily:
            raise ValueError(
                f"Unexpected API response for {loc.name}. Keys: {list(data.keys())}"
            )

        df = pd.DataFrame(daily)
        df.rename(columns={"time": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])

        for var in self.VARIABLES:
            if var not in df.columns:
                logger.warning("[WEATHER] '%s' missing for %s — NaN filled", var, loc.name)
                df[var] = np.nan

        logger.info("[WEATHER] %s — %d rows fetched OK", loc.name, len(df))
        return df[["date"] + self.VARIABLES]


# ---------------------------------------------------------------------------
# Flood / Discharge Fetcher
# ---------------------------------------------------------------------------
class FloodFetcher:
    def __init__(self, config: PipelineConfig, session: ResilientSession):
        self.config = config
        self.session = session

    def fetch(self, loc: LocationConfig) -> pd.DataFrame:
        logger.info("[FLOOD API] Fetching discharge for %s ...", loc.name)
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "start_date": self.config.historical_start,
            "end_date": self.config.historical_end,
            "daily": "river_discharge",
        }
        try:
            data = self.session.get(self.config.flood_api, params)
        except requests.exceptions.RequestException as exc:
            logger.warning("[FLOOD API] Failed for %s — discharge = NaN: %s", loc.name, exc)
            return pd.DataFrame(columns=["date", "river_discharge"])

        daily = data.get("daily", {})
        if not daily or "time" not in daily:
            logger.warning("[FLOOD API] No data returned for %s", loc.name)
            return pd.DataFrame(columns=["date", "river_discharge"])

        df = pd.DataFrame(daily)
        df.rename(columns={"time": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])

        if "river_discharge" not in df.columns:
            df["river_discharge"] = np.nan

        non_null = int(df["river_discharge"].notna().sum())
        logger.info("[FLOOD API] %s — %d rows, %d non-null discharge values", loc.name, len(df), non_null)
        return df[["date", "river_discharge"]]


# ---------------------------------------------------------------------------
# Label Engine
# ---------------------------------------------------------------------------
class LabelEngine:
    FLOOD_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]

    def __init__(self, config: PipelineConfig):
        self.config = config

    def label(self, df: pd.DataFrame, loc: LocationConfig) -> pd.DataFrame:
        df = df.copy()

        if not loc.flood_site:
            df["flood_occurred"] = 0
            logger.info("[LABEL] %s — control site, all labels = 0", loc.name)
            return df

        flood_year = loc.flood_year or 2022
        in_flood_year = df["date"].dt.year.isin([2020, 2021, 2022])
        in_flood_season = df["date"].dt.month.isin(self.FLOOD_SEASON_MONTHS)

        has_discharge = (
            "river_discharge" in df.columns
            and df["river_discharge"].notna().sum() > 30
        )

        if has_discharge:
            p_thresh = np.nanpercentile(
                df["river_discharge"].values,
                self.config.flood_discharge_percentile,
            )
            logger.info("[LABEL] %s discharge %.0f-pct threshold = %.2f m³/s",
                        loc.name, self.config.flood_discharge_percentile, p_thresh)
            discharge_spike = df["river_discharge"] > p_thresh
            df["flood_occurred"] = (
                in_flood_year & in_flood_season & discharge_spike
            ).astype(int)
        else:
            logger.warning("[LABEL] %s — no discharge, using rainfall fallback", loc.name)
            rolling_rain = (
                df["precipitation_sum"].fillna(0)
                .rolling(window=7, min_periods=3)
                .sum()
            )
            heavy_rain = rolling_rain > self.config.flood_rainfall_threshold_mm
            rain_hours = df.get(
                "precipitation_hours", pd.Series(0.0, index=df.index)
            ).fillna(0)
            soil_wet_proxy = rain_hours > 6.0
            df["flood_occurred"] = (
                in_flood_year & in_flood_season & heavy_rain & soil_wet_proxy
            ).astype(int)

        pos = int(df["flood_occurred"].sum())
        total = len(df)
        logger.info("[LABEL] %s — %d/%d flood days (%.1f%%)",
                    loc.name, pos, total, 100 * pos / total if total else 0)
        return df


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------
class FeatureEngineer:
    def engineer(self, df: pd.DataFrame, loc_name: str) -> pd.DataFrame:
        df = df.copy().sort_values("date").reset_index(drop=True)

        rain = df.get("precipitation_sum", pd.Series(0.0, index=df.index)).fillna(0)
        df["rain_3d_sum"] = rain.rolling(3, min_periods=1).sum()
        df["rain_7d_sum"] = rain.rolling(7, min_periods=3).sum()
        df["rain_14d_sum"] = rain.rolling(14, min_periods=7).sum()

        if "river_discharge" in df.columns:
            disch = df["river_discharge"]
            df["discharge_lag1"] = disch.shift(1)
            df["discharge_lag3"] = disch.shift(3)
        else:
            df["discharge_lag1"] = np.nan
            df["discharge_lag3"] = np.nan

        if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
            df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

        logger.info("[FEATURES] %s — temporal features added", loc_name)
        return df


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------
class DataCleaner:
    MAX_NAN_FRACTION = 0.60

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        original_rows = len(df)
        logger.info("[CLEAN] Input: %d rows, %d columns", original_rows, len(df.columns))

        nan_fractions = df.isnull().mean()
        cols_to_drop = [
            c for c in nan_fractions[nan_fractions > self.MAX_NAN_FRACTION].index
            if c not in ("flood_occurred", "date", "location", "state")
        ]
        if cols_to_drop:
            logger.warning("[CLEAN] Dropping high-NaN columns: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)

        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "flood_occurred"
        ]
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        pre_drop = len(df)
        df = df.dropna(subset=["flood_occurred"])
        if len(df) < pre_drop:
            logger.warning("[CLEAN] Dropped %d unlabelled rows", pre_drop - len(df))

        df["flood_occurred"] = df["flood_occurred"].astype(int)
        df["date"] = pd.to_datetime(df["date"])

        remaining_nan = df[numeric_cols].isnull().sum().sum()
        if remaining_nan > 0:
            logger.warning("[CLEAN] %d NaN remain after imputation — filling 0", remaining_nan)
            df[numeric_cols] = df[numeric_cols].fillna(0)

        logger.info("[CLEAN] Output: %d rows, %d columns", len(df), len(df.columns))
        return df


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class NaijaClimaGuardPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.session = ResilientSession(config)
        self.weather_fetcher = WeatherFetcher(config, self.session)
        self.flood_fetcher = FloodFetcher(config, self.session)
        self.label_engine = LabelEngine(config)
        self.feature_engineer = FeatureEngineer()
        self.cleaner = DataCleaner()

    def _process_location(self, loc: LocationConfig) -> Optional[pd.DataFrame]:
        logger.info("=" * 60)
        logger.info("Processing: %s, %s (flood_site=%s)", loc.name, loc.state, loc.flood_site)
        logger.info("=" * 60)
        try:
            weather_df = self.weather_fetcher.fetch(loc)
            time.sleep(1.0)

            discharge_df = self.flood_fetcher.fetch(loc)
            time.sleep(1.0)

            if not discharge_df.empty:
                df = pd.merge(weather_df, discharge_df, on="date", how="left")
            else:
                df = weather_df.copy()
                df["river_discharge"] = np.nan

            df["location"] = loc.name
            df["state"] = loc.state
            df["latitude"] = loc.latitude
            df["longitude"] = loc.longitude
            df["flood_site"] = int(loc.flood_site)

            df = self.label_engine.label(df, loc)
            df = self.feature_engineer.engineer(df, loc.name)
            return df

        except Exception as exc:
            logger.error("FAILED for %s: %s", loc.name, exc, exc_info=True)
            return None

    def run(self) -> pd.DataFrame:
        logger.info("NaijaClimaGuard Pipeline — START")
        frames = []
        for loc in self.config.locations:
            df = self._process_location(loc)
            if df is not None:
                frames.append(df)

        if not frames:
            raise RuntimeError(
                "All locations failed. Check internet connection and try again."
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info("Combined: %d rows from %d/%d locations",
                    len(combined), len(frames), len(self.config.locations))

        combined = self.cleaner.clean(combined)
        self._log_summary(combined)

        combined.to_csv(self.config.output_csv, index=False)
        logger.info("Saved: %s", self.config.output_csv)
        return combined

    def _log_summary(self, df: pd.DataFrame) -> None:
        total = len(df)
        flood_days = int(df["flood_occurred"].sum())
        logger.info("-" * 60)
        logger.info("SUMMARY")
        logger.info("Total rows   : %d", total)
        logger.info("Flood days   : %d (%.1f%%)", flood_days, 100 * flood_days / total)
        logger.info("Non-flood    : %d (%.1f%%)", total - flood_days, 100 * (total - flood_days) / total)
        logger.info("Date range   : %s to %s", df["date"].min().date(), df["date"].max().date())
        logger.info("Columns      : %s", df.columns.tolist())
        for loc_name, grp in df.groupby("location"):
            logger.info("  %-12s  flood_days=%d / %d", loc_name, grp["flood_occurred"].sum(), len(grp))
        logger.info("-" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = PipelineConfig()
    try:
        pipeline = NaijaClimaGuardPipeline(config)
        df = pipeline.run()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Output file    : {config.output_csv}")
        print(f"Total rows     : {len(df)}")
        print(f"Flood days     : {df['flood_occurred'].sum()}")
        print(f"Class balance  : {df['flood_occurred'].mean():.2%} positive")
        print(f"Audit log      : {LOG_FILE}")
        print("=" * 60)

        print("\nSample — Lokoja October 2022:")
        sample = df[
            (df["location"] == "Lokoja") &
            (df["date"].dt.year == 2022) &
            (df["date"].dt.month == 10)
        ][["date", "precipitation_sum", "river_discharge", "rain_7d_sum", "flood_occurred"]]
        if not sample.empty:
            print(sample.to_string(index=False))
        else:
            print("No Lokoja Oct 2022 rows found — check location fetch logs above.")

    except Exception as exc:
        logger.critical("Pipeline aborted: %s", exc, exc_info=True)
        sys.exit(1)