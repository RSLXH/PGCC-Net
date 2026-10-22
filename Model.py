"""PGCC-Net model, dataset, physical descriptors, and loss function."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "model"
PREPROCESSING_SCHEMA = "pgcc_net_v5_strict_preprocessing"

CONFIG = {
    "IMG_PATH": str(DATA_DIR / "Combined_Images.npy"),
    "ANG_PATH": str(DATA_DIR / "Combined_Angles.npy"),
    "LBL_PATH": str(DATA_DIR / "Combined_Metadata.csv"),
    "TRAIN_INDEX_PATH": str(OUTPUT_DIR / "train_indices.npy"),
    "STATS_PATH": str(OUTPUT_DIR / "stats.json"),
    "CI_BG_PATH": str(OUTPUT_DIR / "ci_background.npz"),
    "FEATURE_CACHE_PREFIX": str(OUTPUT_DIR / "feature_cache"),
    "BATCH_SIZE": 256,
    "LEARNING_RATE": 0.001,
    "EPOCHS": 60,
    "RANDOM_SEED": 42,
    "ADAPTIVE_LOG_VARS_INIT": [0.0, 0.0, 0.0, 0.0],
    "DIRECT_TAU_MAX": 1.10,
    "CI_BAND_INDEX": 2,
    "CI_LOW_PERCENTILE": 5.0,
    "CI_HIGH_PERCENTILE": 95.0,
    "CI_HIGH_THRESHOLD": 0.8,
    "CI_BACKGROUND_MODE": "latband_month_localhour",
    "CI_GROUP_MIN_SAMPLES": 30,
    "CI_LAT_BAND_WIDTH_DEG": 10.0,
    "CI_LOCAL_HOUR_BIN": 2,
    "USE_FEATURE_CACHE": True,
    "NUM_WORKERS": 0,
    "PREFETCH_FACTOR": 2,
    "USE_AMP": True,
    "CHANNELS_LAST": True,
    "GATE_GAMMA": 0.30,
    "SOLAR_CONSTANT": 1367.0,
    "RTOA_SCALE": 1400.0,
    "PATCH_SIZE": 16,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model-development stations from Supplementary Table S1. Only the training
# subset of their 2017/2019/2020 observations is used for parameter estimation.
TRAINING_STATIONS = (
    "BOM_SL003003", "BOM_SL012038", "BOM_SL023034", "BOM_SL032040",
    "BOM_SL039083", "BOM_SL086282", "BOM_SL008315", "BOM_SL072150",
    "BSRN_FUA", "BSRN_HOW", "BSRN_ISH", "BSRN_MNM", "BSRN_TAT", "BSRN_YUS",
)

# Spatially independent stations from Supplementary Table S2.
SPATIAL_TEST_STATIONS = (
    "BOM_SL005007", "BOM_SL091148",
    "BSRN_ASP", "BSRN_COC", "BSRN_DWN", "BSRN_KWA", "BSRN_LAU",
    "BSRN_LYU", "BSRN_NEW", "BSRN_SAP",
)

DEVELOPMENT_YEARS = (2017, 2019, 2020)
TEMPORAL_TEST_YEARS = (2016, 2018)
TRAIN_FRACTION = 0.7

# Keep the CSV reader's column aliases explicit. These constants were missing
# even though read_model_metadata() already referenced them.
MODEL_METADATA_CATEGORY_COLUMNS = frozenset({
    "Network_Station_ID", "Station_ID", "station_id", "station", "Station",
    "site", "Site", "SITE", "BSRN_ID", "bsrn_id", "Site Number", "site_number",
    "id", "ID", "network", "Network", "source", "Source", "is_daytime",
})
MODEL_METADATA_FLOAT32_COLUMNS = frozenset({
    "global2_mean", "global_mean", "Rg", "R_g", "global", "Global",
    "direct_mean_H", "direct_H", "direct_mean", "Rdir", "R_dir", "direct", "Direct",
    "diffuse_mean", "Rdif", "R_dif", "diffuse", "Diffuse",
    "lat", "Lat", "latitude", "Latitude", "LAT",
    "lon", "Lon", "longitude", "Longitude", "LON",
    "SOZ", "soz", "ground_solar_zenith_mean", "evaluation_solar_zenith_deg",
})
MODEL_METADATA_COLUMN_CANDIDATES = (
    MODEL_METADATA_CATEGORY_COLUMNS | MODEL_METADATA_FLOAT32_COLUMNS | frozenset({
        "DOY", "doy", "dayofyear", "DayOfYear", "year", "Year", "YEAR", "yy",
        "UTC", "utc", "datetime", "date_time", "time_utc", "DateTime",
        "Timestamp", "timestamp", "date", "mm", "dd", "month", "Month",
        "hour", "Hour", "HH", "time", "Time", "day", "Day",
    })
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_linear_layers(module):
    """Initialize fully connected layers with Xavier uniform weights."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def initialize_convolution_layers(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


def clean_path(path: str) -> str:
    'Data processing note.'
    return path.strip() if isinstance(path, str) else path


def get_feature_cache_paths(prefix: str):
    prefix = clean_path(prefix)
    return {
        "ang": prefix + "_ang.npy",
        "phys": prefix + "_phys.npy",
        "y": prefix + "_y.npy",
        "meta": prefix + "_meta.json",
    }


def standardize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    # The metadata table can exceed 1 GB.  It is always newly loaded before
    # this helper is called, so renaming in place avoids a second full copy.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_first_column(df: pd.DataFrame, candidates, required_name: str):
    'Data processing note.'
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError('Data processing note.')


def get_label_columns(df: pd.DataFrame):
    col_g = find_first_column(
        df,
        candidates=["global2_mean", "global_mean", "Rg", "R_g", "global", "Global"],
        required_name="global radiation"
    )
    col_dir = find_first_column(
        df,
        candidates=["direct_mean_H", "direct_H", "direct_mean", "Rdir", "R_dir", "direct", "Direct"],
        required_name="direct horizontal radiation"
    )
    col_dif = find_first_column(
        df,
        candidates=["diffuse_mean", "Rdif", "R_dif", "diffuse", "Diffuse"],
        required_name="diffuse radiation"
    )
    return col_g, col_dir, col_dif


def read_model_metadata(path: str) -> pd.DataFrame:
    """Read only model-relevant CSV columns using memory-efficient dtypes."""
    path = clean_path(path)
    raw_columns = list(pd.read_csv(path, nrows=0).columns)
    candidate_set = set(MODEL_METADATA_COLUMN_CANDIDATES)
    selected_raw = [
        raw for raw in raw_columns if str(raw).strip() in candidate_set
    ]
    if not selected_raw:
        raise ValueError("No model-relevant columns were found in metadata: {}".format(path))

    dtype_map = {}
    for raw in selected_raw:
        name = str(raw).strip()
        if name in MODEL_METADATA_FLOAT32_COLUMNS:
            dtype_map[raw] = np.float32
        elif name in MODEL_METADATA_CATEGORY_COLUMNS:
            dtype_map[raw] = "category"

    df = pd.read_csv(
        path,
        usecols=selected_raw,
        dtype=dtype_map,
        # Let the C parser process the 1+ GB source in internal chunks.  With
        # low_memory=False it may retain a very large token buffer even though
        # usecols selects only a small subset of fields.
        low_memory=True,
    )
    df = standardize_df_columns(df)
    print(">>> Metadata: loaded {} required columns out of {} ({} rows)".format(
        len(df.columns), len(raw_columns), len(df)
    ))
    return df


def infer_doy_from_df(df: pd.DataFrame, n: int) -> np.ndarray:
    """Use observed DOY/date for Eq. (6); missing dates must not become day 172."""
    for col in ["DOY", "doy", "dayofyear", "DayOfYear"]:
        if col in df.columns:
            doy = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if (len(doy) != n or not np.isfinite(doy).all()
                    or not ((doy >= 1) & (doy <= 366)).all()
                    or not np.equal(doy, np.floor(doy)).all()):
                raise ValueError("DOY must contain valid integer days 1-366 for every sample")
            return doy.astype(np.float32)
    dates = observation_dates(df)
    if dates is None or dates.isna().any() or len(dates) != n:
        raise ValueError("Provide complete observed DOY or dates; missing dates cannot default to day 172")
    return dates.dt.dayofyear.to_numpy(dtype=np.float32)


def observation_dates(df):
    """Parse supported full UTC date fields or explicit calendar components."""
    for col in ("UTC", "utc", "datetime", "date_time", "time_utc",
                "DateTime", "Timestamp", "timestamp", "date"):
        if col in df.columns:
            return pd.to_datetime(df[col].astype(str), errors="coerce", utc=True)
    year = find_optional_column(df, ("year", "Year", "YEAR", "yy"))
    month = find_optional_column(df, ("mm", "month", "Month"))
    day = find_optional_column(df, ("dd", "day", "Day"))
    if year and month and day:
        years = pd.to_numeric(df[year], errors="coerce")
        if year == "yy":
            years = years.where(~years.between(0, 99), years + 2000)
        return pd.to_datetime({"year": years,
                               "month": pd.to_numeric(df[month], errors="coerce"),
                               "day": pd.to_numeric(df[day], errors="coerce")}, errors="coerce", utc=True)
    return None


def earth_sun_distance_factor(doy: np.ndarray) -> np.ndarray:
    'Data processing note.'
    return 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)


def safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, q))


def find_optional_column(df: pd.DataFrame, candidates):
    'Data processing note.'
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_station_ids_from_df(df: pd.DataFrame, n: int) -> np.ndarray:
    'Data processing note.'
    station_col = find_optional_column(
        df,
        [
            "Network_Station_ID", "Station_ID", "station_id", "station", "Station", "site", "Site", "SITE",
            "BSRN_ID", "bsrn_id", "Site Number", "site_number", "id", "ID"
        ],
    )
    if station_col is not None:
        return df[station_col].astype(str).fillna("UNKNOWN").values

    lon_col = find_optional_column(df, ["lon", "Lon", "longitude", "Longitude", "LON"])
    lat_col = find_optional_column(df, ["lat", "Lat", "latitude", "Latitude", "LAT"])
    if lon_col is not None and lat_col is not None:
        lon = pd.to_numeric(df[lon_col], errors="coerce").round(3).astype(str)
        lat = pd.to_numeric(df[lat_col], errors="coerce").round(3).astype(str)
        return ("lat" + lat + "_lon" + lon).values

    print('Data processing note.')
    return np.array(["ALL"] * n, dtype=object)


def _parse_hour_from_time_series(s: pd.Series) -> pd.Series:
    """Handle UTC hour, HHMM, HHMMSS, and clock strings without guessing dates."""
    numeric = pd.to_numeric(s, errors="coerce")
    hour = pd.Series(np.nan, index=s.index, dtype=float)
    plain = numeric.between(0, 23)
    hour.loc[plain] = np.floor(numeric.loc[plain])
    hhmm = numeric.between(100, 2359) & numeric.eq(np.floor(numeric)) & ((numeric % 100) < 60)
    hour.loc[hhmm] = np.floor(numeric.loc[hhmm] / 100)
    hhmmss = (numeric.between(10000, 235959) & numeric.eq(np.floor(numeric))
              & ((numeric // 100) % 100 < 60) & ((numeric % 100) < 60))
    hour.loc[hhmmss] = np.floor(numeric.loc[hhmmss] / 10000)
    clock = s.astype(str).str.match(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$")
    parsed = pd.to_datetime("2000-01-01 " + s.astype(str).where(clock), errors="coerce", utc=True)
    hour.loc[clock] = parsed.loc[clock].dt.hour
    return hour


def infer_month_hour_from_df(df: pd.DataFrame, n: int):
    'Data processing note.'
    dt_col = find_optional_column(
        df,
        ["UTC", "utc", "datetime", "date_time", "time_utc", "DateTime", "Timestamp", "timestamp"],
    )
    dates = observation_dates(df)
    if dt_col is not None and dates is not None:
        month, hour = dates.dt.month, dates.dt.hour
    else:
        month_col = find_optional_column(df, ("mm", "month", "Month"))
        month = (pd.to_numeric(df[month_col], errors="coerce") if month_col
                 else dates.dt.month if dates is not None else pd.Series(np.nan, index=df.index))
        hour_col = find_optional_column(df, ["hour", "Hour", "HH", "time", "Time"])
        hour = (_parse_hour_from_time_series(df[hour_col]) if hour_col
                else pd.Series(np.nan, index=df.index))
    if (len(month) != n or len(hour) != n or not month.between(1, 12).all()
            or not month.eq(np.floor(month)).all() or not hour.between(0, 23).all()):
        raise ValueError("CI backgrounds require valid observed months and UTC hours for every sample")
    return month.values.astype(int), hour.values.astype(int)


def infer_lat_lon_from_df(df: pd.DataFrame, n: int):
    lat_col = find_optional_column(df, ["lat", "Lat", "latitude", "Latitude", "LAT"])
    lon_col = find_optional_column(df, ["lon", "Lon", "longitude", "Longitude", "LON"])
    if lat_col is None or lon_col is None:
        raise ValueError('Data processing note.')
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=np.float32)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=np.float32)
    if len(lat) != n or len(lon) != n or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('Data processing note.')
    return lat, lon


def infer_network_from_df(df: pd.DataFrame, n: int):
    network_col = find_optional_column(df, ["network", "Network", "source", "Source"])
    if network_col is not None:
        return df[network_col].astype(str).str.upper().fillna("UNKNOWN").to_numpy()
    stations = infer_station_ids_from_df(df, n).astype(str)
    return np.array([x.split("_", 1)[0].upper() if "_" in x else "UNKNOWN" for x in stations])


def make_lat_band(lat, width):
    width = float(width)
    return float(np.floor(float(lat) / width) * width + width / 2.0)


def make_local_hour(utc_hour, lon, bin_width):
    local = (float(utc_hour) + float(lon) / 15.0) % 24.0
    return int(np.floor(local / int(bin_width)) * int(bin_width))


def make_key_lat_month_localhour(lat_band, month, local_hour) -> str:
    return f"lat{float(lat_band):+06.1f}|m{int(month):02d}|lh{int(local_hour):02d}"


def make_key_month_localhour(month, local_hour) -> str:
    return f"m{int(month):02d}|lh{int(local_hour):02d}"


def compute_latband_ci_backgrounds(imgs, ci_band: int, latitudes: np.ndarray,
                                   longitudes: np.ndarray, months: np.ndarray,
                                   utc_hours: np.ndarray, low_q: float,
                                   high_q: float, min_samples: int,
                                   lat_width: float, local_hour_bin: int,
                                   save_path: str, sample_indices: np.ndarray):
    'Data processing note.'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    latitudes = np.asarray(latitudes, dtype=np.float32)[sample_indices]
    longitudes = np.asarray(longitudes, dtype=np.float32)[sample_indices]
    months = np.asarray(months).astype(int)[sample_indices]
    utc_hours = np.asarray(utc_hours).astype(int)[sample_indices]
    n = len(sample_indices)

    lat_bands = np.array([make_lat_band(x, lat_width) for x in latitudes], dtype=np.float32)
    local_hours = np.array([
        make_local_hour(utc_hours[i], longitudes[i], local_hour_bin) for i in range(n)
    ], dtype=np.int16)
    keys_lat = np.array([
        make_key_lat_month_localhour(lat_bands[i], months[i], local_hours[i]) for i in range(n)
    ], dtype=object)
    keys_mh = np.array([
        make_key_month_localhour(months[i], local_hours[i]) for i in range(n)
    ], dtype=object)

    def build_level(keys, level_name):
        grouped_positions = {}
        for position, key in enumerate(keys):
            grouped_positions.setdefault(str(key), []).append(position)
        out_keys, lows, highs, counts = [], [], [], []
        for k, positions in tqdm(
            grouped_positions.items(), desc=f"CI background {level_name}", leave=False
        ):
            local_idx = np.asarray(positions, dtype=np.int64)
            if len(local_idx) < min_samples:
                continue
            source_idx = sample_indices[local_idx]
            arr = np.array(imgs[source_idx, ci_band, :, :], dtype=np.float32)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            if np.all(np.isnan(arr)):
                continue
            low = np.nanpercentile(arr, low_q, axis=0).astype(np.float32)
            high = np.nanpercentile(arr, high_q, axis=0).astype(np.float32)
            out_keys.append(str(k)); lows.append(low); highs.append(high); counts.append(len(local_idx))
        if len(out_keys) == 0:
            sample = np.array(imgs[0, ci_band, :, :], dtype=np.float32)
            h, w = sample.shape
            return (np.array([], dtype=str), np.empty((0, h, w), dtype=np.float32),
                    np.empty((0, h, w), dtype=np.float32), np.array([], dtype=np.int32))
        return (np.array(out_keys, dtype=str), np.stack(lows), np.stack(highs), np.array(counts, dtype=np.int32))

    lat_keys, lat_low, lat_high, lat_count = build_level(keys_lat, "latband-month-localhour")
    mh_keys, mh_low, mh_high, mh_count = build_level(keys_mh, "month-localhour")

    arr_all = np.array(imgs[sample_indices, ci_band, :, :], dtype=np.float32)
    arr_all = np.where(np.isfinite(arr_all), arr_all, np.nan)
    g_low = np.nanpercentile(arr_all, low_q, axis=0).astype(np.float32)
    g_high = np.nanpercentile(arr_all, high_q, axis=0).astype(np.float32)

    np.savez_compressed(
        save_path,
        lat_keys=lat_keys, lat_low=lat_low, lat_high=lat_high, lat_count=lat_count,
        mh_keys=mh_keys, mh_low=mh_low, mh_high=mh_high, mh_count=mh_count,
        lat_width=np.array([lat_width], dtype=np.float32),
        local_hour_bin=np.array([local_hour_bin], dtype=np.int16),
        global_low=g_low, global_high=g_high,
    )
    print('Data processing note.')
    print(f"    latband-month-localhour groups: {len(lat_keys)}")
    print(f"    month-localhour groups:         {len(mh_keys)}")


def load_index_file(path):
    values = np.load(clean_path(path))
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError('Data processing note.')
    return values


def station_indices(metadata, stations, daytime_only=False):
    """Return row indices for one station group."""
    station_ids = infer_station_ids_from_df(metadata, len(metadata)).astype(str)
    stations = tuple(str(value) for value in stations)
    mask = np.isin(station_ids, stations)
    present = set(station_ids[mask].tolist())
    missing = sorted(set(stations) - present)
    if missing:
        raise ValueError("The dataset is missing stations: {}".format(missing))
    if daytime_only:
        if "is_daytime" not in metadata.columns:
            raise KeyError("Daytime filtering requires the is_daytime column")
        daytime = metadata["is_daytime"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        mask &= daytime
    return np.flatnonzero(mask).astype(np.int64)


def index_fingerprint(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def infer_years_from_df(metadata):
    """Read observed calendar years; never guess a missing year for a split."""
    for column in ("year", "Year", "YEAR", "yy"):
        if column in metadata.columns:
            years = pd.to_numeric(metadata[column], errors="coerce").to_numpy(dtype=float)
            if column == "yy":
                years = np.where((years >= 0) & (years <= 99), 2000 + years, years)
            if (np.isfinite(years).all() and np.equal(years, np.floor(years)).all()
                    and ((years >= 1900) & (years <= 2100)).all()):
                return years.astype(np.int64)

    for column in ("UTC", "utc", "datetime", "date_time", "time_utc",
                   "DateTime", "Timestamp", "timestamp", "date"):
        if column in metadata.columns:
            # Numeric hours must not silently become nanoseconds since 1970.
            dates = pd.to_datetime(metadata[column].astype(str), errors="coerce", utc=True)
            if dates.notna().all():
                return dates.dt.year.to_numpy(dtype=np.int64)
    raise ValueError(
        "Calendar years are required to enforce the manuscript split. Supply "
        "a valid year/yy column or a complete date-time column; DOY alone is insufficient."
    )


def manuscript_split_indices(metadata, seed=42):
    """Build the manuscript's disjoint development/spatial/temporal subsets.

    The seed provides a reproducible new split; it cannot recover the exact
    historical split without the original experiment's saved row indices.
    """
    stations = infer_station_ids_from_df(metadata, len(metadata)).astype(str)
    development = set(TRAINING_STATIONS)
    spatial = set(SPATIAL_TEST_STATIONS)
    if len(development) != 14 or len(spatial) != 10 or development & spatial:
        raise ValueError("Expected 14 development and 10 disjoint spatial-test stations")
    unknown = set(stations.tolist()) - (development | spatial)
    if unknown:
        raise ValueError("Stations outside Supplementary Tables S1/S2: {}".format(sorted(unknown)))
    years = infer_years_from_df(metadata)
    unsupported = set(np.unique(years).tolist()) - set(DEVELOPMENT_YEARS + TEMPORAL_TEST_YEARS)
    if unsupported:
        raise ValueError("Years outside the manuscript's 2016-2020 study period: {}".format(sorted(unsupported)))
    development_mask = np.isin(stations, TRAINING_STATIONS) & np.isin(years, DEVELOPMENT_YEARS)
    present = set(stations[development_mask].tolist())
    if present != development:
        raise ValueError("Missing development stations in 2017/2019/2020: {}".format(sorted(development - present)))
    development_indices = np.flatnonzero(development_mask)
    shuffled = np.random.RandomState(int(seed)).permutation(development_indices)
    # Match a 30% validation split, rounded upward for indivisible sample counts.
    n_validation = int(np.ceil((1.0 - TRAIN_FRACTION) * len(shuffled) - 1e-12))
    if n_validation == 0 or n_validation >= len(shuffled):
        raise ValueError("Insufficient development observations for a 7:3 split")
    subsets = {
        "train": np.sort(shuffled[n_validation:]).astype(np.int64),
        "validation": np.sort(shuffled[:n_validation]).astype(np.int64),
        "spatial_test": np.flatnonzero(np.isin(stations, SPATIAL_TEST_STATIONS)
                                      & np.isin(years, DEVELOPMENT_YEARS)).astype(np.int64),
        "temporal_test": np.flatnonzero(np.isin(years, TEMPORAL_TEST_YEARS)).astype(np.int64),
    }
    if set(stations[subsets["train"]].tolist()) != development:
        raise ValueError("Random training subset omits a development station; check sample counts")
    return subsets


def file_signature(path):
    path = Path(clean_path(path)).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def preprocessing_contract():
    keys = ("SOLAR_CONSTANT", "RTOA_SCALE", "CI_BAND_INDEX", "CI_LOW_PERCENTILE",
            "CI_HIGH_PERCENTILE", "CI_HIGH_THRESHOLD", "CI_BACKGROUND_MODE",
            "CI_GROUP_MIN_SAMPLES", "CI_LAT_BAND_WIDTH_DEG", "CI_LOCAL_HOUR_BIN", "PATCH_SIZE")
    return {"schema": PREPROCESSING_SCHEMA, "settings": {key: CONFIG[key] for key in keys}}


def preprocessing_artifact_hashes():
    """Bind checkpoints to the normalization/CI artifacts used to train them."""
    with open(CONFIG["STATS_PATH"], encoding="utf-8") as handle:
        stats = json.load(handle)
    hashes = {"stats": hashlib.sha256(Path(CONFIG["STATS_PATH"]).read_bytes()).hexdigest()}
    if stats["ci_background_mode"] == "latband_month_localhour":
        hashes["ci_background"] = hashlib.sha256(Path(CONFIG["CI_BG_PATH"]).read_bytes()).hexdigest()
    return hashes


def training_statistics(images, labels, indices, batch_size=1024):
    """Compute population moments of every training sample in bounded memory."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("Training indices are empty")
    image_count = label_count = 0
    image_mean = image_m2 = label_mean = label_m2 = None
    def merge(count, mean, m2, values):
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        if not count:
            return batch_count, batch_mean, batch_m2
        delta = batch_mean - mean
        combined = count + batch_count
        return (combined, mean + delta * batch_count / combined,
                m2 + batch_m2 + delta ** 2 * count * batch_count / combined)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        image_batch = np.asarray(images[selected], dtype=np.float64)
        label_batch = np.asarray(labels[selected], dtype=np.float64)
        if not np.isfinite(image_batch).all() or not np.isfinite(label_batch).all():
            raise ValueError("Training data contain invalid reflectances or labels; apply upstream quality control")
        pixels = image_batch.transpose(0, 2, 3, 1).reshape(-1, image_batch.shape[1])
        image_count, image_mean, image_m2 = merge(image_count, image_mean, image_m2, pixels)
        label_count, label_mean, label_m2 = merge(label_count, label_mean, label_m2, label_batch)
    image_std = np.sqrt(image_m2 / image_count)
    label_std = np.sqrt(label_m2 / label_count)
    # Constant columns have no meaningful scale; keep the prior unit-scale fallback.
    return image_mean, np.where(image_std < 1e-6, 1.0, image_std), label_mean, np.where(label_std < 1e-6, 1.0, label_std)


def ensure_stats_exist(metadata=None):
    train_indices = load_index_file(CONFIG["TRAIN_INDEX_PATH"])
    train_signature = index_fingerprint(train_indices)
    data_signature = {key: file_signature(CONFIG[key]) for key in ("IMG_PATH", "ANG_PATH", "LBL_PATH")}
    required_keys = {
        "img_mean", "img_std", "lbl_mean", "lbl_std", "ci_background_mode",
        "ci_band_index", "train_index_fingerprint", "ci_lat_band_width_deg",
        "ci_local_hour_bin", "preprocessing_contract", "data_signature",
    }
    stats_exists = os.path.exists(CONFIG["STATS_PATH"])
    bg_required = CONFIG.get("CI_BACKGROUND_MODE", "global") == "latband_month_localhour"
    bg_exists = os.path.exists(clean_path(CONFIG.get("CI_BG_PATH", ""))) if bg_required else True

    if stats_exists:
        with open(CONFIG["STATS_PATH"], "r") as f:
            stats = json.load(f)
        if (required_keys.issubset(set(stats.keys())) and bg_exists
                and stats.get("train_index_fingerprint") == train_signature
                and stats.get("preprocessing_contract") == preprocessing_contract()
                and stats.get("data_signature") == data_signature
                and stats.get("ci_background_mode") == CONFIG.get("CI_BACKGROUND_MODE")
                and float(stats.get("ci_lat_band_width_deg")) == float(CONFIG["CI_LAT_BAND_WIDTH_DEG"])
                and int(stats.get("ci_local_hour_bin")) == int(CONFIG["CI_LOCAL_HOUR_BIN"])):
            print('Data processing note.')
            if bg_required:
                print('Data processing note.')
            return
        else:
            print('Data processing note.')

    print('Data processing note.')
    imgs = np.load(clean_path(CONFIG["IMG_PATH"]), mmap_mode="r")
    df = metadata if metadata is not None else read_model_metadata(CONFIG["LBL_PATH"])
    col_g, col_dir, col_dif = get_label_columns(df)
    lbls = df[[col_g, col_dir, col_dif]].values.astype(np.float32)

    img_mean, img_std, lbl_mean, lbl_std = training_statistics(imgs, lbls, train_indices)

    ci_band = int(CONFIG["CI_BAND_INDEX"])
    ci_band = max(0, min(ci_band, imgs.shape[1] - 1))

    stats = {
        "img_mean": img_mean.tolist(),
        "img_std": img_std.tolist(),
        "lbl_mean": lbl_mean.tolist(),
        "lbl_std": lbl_std.tolist(),
        "label_columns": [col_g, col_dir, col_dif],
        "ci_band_index": ci_band,
        "ci_background_mode": CONFIG.get("CI_BACKGROUND_MODE", "latband_month_localhour"),
        "ci_low_percentile": CONFIG["CI_LOW_PERCENTILE"],
        "ci_high_percentile": CONFIG["CI_HIGH_PERCENTILE"],
        "ci_lat_band_width_deg": float(CONFIG["CI_LAT_BAND_WIDTH_DEG"]),
        "ci_local_hour_bin": int(CONFIG["CI_LOCAL_HOUR_BIN"]),
        "train_index_fingerprint": train_signature,
        "train_sample_count": int(len(train_indices)),
        "normalization_sample_count": int(len(train_indices)),
        "standard_deviation_ddof": 0,
        "preprocessing_contract": preprocessing_contract(),
        "data_signature": data_signature,
    }

    if CONFIG.get("CI_BACKGROUND_MODE") == "latband_month_localhour":
        latitudes, longitudes = infer_lat_lon_from_df(df, len(df))
        months, hours = infer_month_hour_from_df(df, len(df))
        # Use the same central-pixel satellite geometry as the model input.
        angles = np.load(clean_path(CONFIG["ANG_PATH"]), mmap_mode="r")
        daytime = np.asarray(angles[:, 2]) < 90.0
        ci_train_indices = train_indices[daytime[train_indices]]
        if len(ci_train_indices) < int(CONFIG["CI_GROUP_MIN_SAMPLES"]):
            raise ValueError('Data processing note.')
        bg_path = clean_path(CONFIG["CI_BG_PATH"])
        compute_latband_ci_backgrounds(
            imgs=imgs,
            ci_band=ci_band,
            latitudes=latitudes,
            longitudes=longitudes,
            months=months,
            utc_hours=hours,
            low_q=CONFIG["CI_LOW_PERCENTILE"],
            high_q=CONFIG["CI_HIGH_PERCENTILE"],
            min_samples=int(CONFIG["CI_GROUP_MIN_SAMPLES"]),
            lat_width=float(CONFIG["CI_LAT_BAND_WIDTH_DEG"]),
            local_hour_bin=int(CONFIG["CI_LOCAL_HOUR_BIN"]),
            save_path=bg_path,
            sample_indices=ci_train_indices,
        )
        stats["ci_bg_path"] = bg_path
        stats["ci_group_min_samples"] = int(CONFIG["CI_GROUP_MIN_SAMPLES"])
        stats["ci_background_sample_count"] = int(len(ci_train_indices))
        stats["ci_clear"] = None
        stats["ci_cloud"] = None
    else:
        s_img_for_ci = imgs[train_indices].astype(np.float32)
        vis_values = s_img_for_ci[:, ci_band, :, :].reshape(-1)
        ci_clear = safe_percentile(vis_values, CONFIG["CI_LOW_PERCENTILE"], default=0.0)
        ci_cloud = safe_percentile(vis_values, CONFIG["CI_HIGH_PERCENTILE"], default=ci_clear + 1.0)
        stats["ci_clear"] = ci_clear
        stats["ci_cloud"] = ci_cloud

    with open(CONFIG["STATS_PATH"], "w") as f:
        json.dump(stats, f, indent=2)
    print('Data processing note.')
    print(f"    CI band index = {ci_band}")
    print(f"    CI background mode = {stats['ci_background_mode']}")


class NativeDataset(Dataset):
    def __init__(self, img_path, ang_path, lbl_path, stats_path, metadata=None,
                 fit_stats=True, require_labels=True):
        self.images = np.load(clean_path(img_path), mmap_mode="r")
        self.angles = np.load(clean_path(ang_path), mmap_mode="r")
        df = metadata if metadata is not None else read_model_metadata(lbl_path)
        patch = int(CONFIG["PATCH_SIZE"])
        if (self.images.ndim != 4 or tuple(self.images.shape[1:]) != (6, patch, patch)
                or self.angles.shape != (len(self.images), 4) or len(df) != len(self.images)):
            raise ValueError("Expected aligned images (N,6,{0},{0}), native angles (N,4), and N metadata rows".format(patch))
        if not np.isfinite(self.angles).all():
            raise ValueError("Satellite angles contain missing or non-finite values")
        # Native angle order is [VZA, VAA, SZA, SAA], all in degrees.
        if (not ((self.angles[:, 0] >= 0) & (self.angles[:, 0] <= 180)).all()
                or not ((self.angles[:, 2] >= 0) & (self.angles[:, 2] <= 180)).all()):
            raise ValueError("Solar and viewing zenith angles must be in degrees within [0,180]")
        self.labels = None
        if require_labels:
            col_g, col_dir, col_dif = get_label_columns(df)
            self.labels = df[[col_g, col_dir, col_dif]].values.astype(np.float32)
            if not np.isfinite(self.labels).all():
                raise ValueError("Observed radiation labels contain missing or non-finite values")
        if fit_stats:
            if not require_labels:
                raise ValueError("Fitting preprocessing requires training labels and saved training indices")
            ensure_stats_exist(metadata=df)
        with open(stats_path, "r") as f:
            stats = json.load(f)
        if stats.get("preprocessing_contract") != preprocessing_contract():
            raise ValueError("Normalization artifacts use a different preprocessing contract")

        self.img_mean = np.array(stats["img_mean"]).reshape(6, 1, 1).astype(np.float32)
        self.img_std = np.array(stats["img_std"]).reshape(6, 1, 1).astype(np.float32)
        self.lbl_mean = np.array(stats["lbl_mean"]).astype(np.float32)
        self.lbl_std = np.array(stats["lbl_std"]).astype(np.float32)

        self.ci_band_index = int(stats.get("ci_band_index", CONFIG["CI_BAND_INDEX"]))
        self.ci_background_mode = stats.get("ci_background_mode", CONFIG.get("CI_BACKGROUND_MODE", "latband_month_localhour"))
        self.ci_clear = stats.get("ci_clear", 0.0)
        self.ci_cloud = stats.get("ci_cloud", 1.0)

        self.doy = infer_doy_from_df(df, len(df)).astype(np.float32)

        self.station_ids = infer_station_ids_from_df(df, len(df)).astype(str)
        self.latitudes, self.longitudes = infer_lat_lon_from_df(df, len(df))
        self.months, self.hours = infer_month_hour_from_df(df, len(df))
        self.months = self.months.astype(int)
        self.hours = self.hours.astype(int)
        self.lat_bands = np.array([
            make_lat_band(x, CONFIG["CI_LAT_BAND_WIDTH_DEG"]) for x in self.latitudes
        ], dtype=np.float32)
        self.local_hours = np.array([
            make_local_hour(self.hours[i], self.longitudes[i], CONFIG["CI_LOCAL_HOUR_BIN"])
            for i in range(len(df))
        ], dtype=np.int16)

        self.ci_bg = None
        self.ci_lat_map = {}
        self.ci_lat_candidates = {}
        self.ci_mh_map = {}
        if self.ci_background_mode == "latband_month_localhour":
            # Model folders may be renamed after training.  Prefer the recorded
            # path when it still exists; otherwise safely fall back to the
            # current runtime configuration supplied by the training/inference
            # script, without regenerating the CI background.
            recorded_bg_path = clean_path(stats.get("ci_bg_path", ""))
            configured_bg_path = clean_path(CONFIG.get("CI_BG_PATH", ""))
            if configured_bg_path and os.path.exists(configured_bg_path):
                bg_path = configured_bg_path
            elif recorded_bg_path and os.path.exists(recorded_bg_path):
                bg_path = recorded_bg_path
            else:
                bg_path = recorded_bg_path or configured_bg_path
            if not bg_path or not os.path.exists(bg_path):
                raise FileNotFoundError(f"CI background file not found: {bg_path}")
            bg = np.load(bg_path, allow_pickle=False)
            lat_keys = bg["lat_keys"].astype(str)
            self.ci_lat_low = bg["lat_low"].astype(np.float32)
            self.ci_lat_high = bg["lat_high"].astype(np.float32)
            for i, key in enumerate(lat_keys):
                self.ci_lat_map[str(key)] = i
                parts = str(key).split("|")
                lat_value = float(parts[0].replace("lat", ""))
                month_value = int(parts[1].replace("m", ""))
                hour_value = int(parts[2].replace("lh", ""))
                self.ci_lat_candidates.setdefault((month_value, hour_value), []).append((lat_value, i))
            mh_keys = bg["mh_keys"].astype(str)
            self.ci_mh_low = bg["mh_low"].astype(np.float32)
            self.ci_mh_high = bg["mh_high"].astype(np.float32)
            self.ci_mh_map = {str(k): i for i, k in enumerate(mh_keys)}
            self.global_ci_low = bg["global_low"].astype(np.float32)
            self.global_ci_high = bg["global_high"].astype(np.float32)
            bg.close()
        else:
            self.ci_clear = float(self.ci_clear)
            self.ci_cloud = float(self.ci_cloud)

        self.soz = np.asarray(self.angles[:, 2], dtype=np.float32)

        # Inference must never fit statistics or reuse a training-row cache.
        self.use_feature_cache = bool(fit_stats and require_labels and CONFIG.get("USE_FEATURE_CACHE", True))
        self.ang_cache = None
        self.phys_cache = None
        self.y_cache = None
        if self.use_feature_cache:
            self._ensure_feature_cache()
            cache_paths = get_feature_cache_paths(CONFIG["FEATURE_CACHE_PREFIX"])
            self.ang_cache = np.load(cache_paths["ang"], mmap_mode="r")
            self.phys_cache = np.load(cache_paths["phys"], mmap_mode="r")
            self.y_cache = np.load(cache_paths["y"], mmap_mode="r")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def process_angles(ang_arr):
        """Native [VZA,VAA,SZA,SAA] degrees -> manuscript's three cosines."""
        rad = np.deg2rad(ang_arr.astype(np.float32))
        vza, vaa, sza, saa = rad[0], rad[1], rad[2], rad[3]
        mu0 = np.cos(sza)
        vza_cos = np.cos(vza)
        raa = np.arccos(np.clip(np.cos(saa - vaa), -1.0, 1.0))
        return np.array([
            mu0,
            vza_cos,
            np.cos(raa),
        ], dtype=np.float32)

    def get_ci_background_for_sample(self, idx: int):
        'Data processing note.'
        if self.ci_background_mode != "latband_month_localhour":
            return self.ci_clear, self.ci_cloud

        lat_band = float(self.lat_bands[idx])
        month = self.months[idx]
        local_hour = int(self.local_hours[idx])

        key_lat = make_key_lat_month_localhour(lat_band, month, local_hour)
        if key_lat in self.ci_lat_map:
            i = self.ci_lat_map[key_lat]
            return self.ci_lat_low[i], self.ci_lat_high[i]

        candidates = self.ci_lat_candidates.get((month, local_hour), [])
        if candidates:
            _, i = min(candidates, key=lambda item: abs(item[0] - lat_band))
            return self.ci_lat_low[i], self.ci_lat_high[i]

        key_mh = make_key_month_localhour(month, local_hour)
        if key_mh in self.ci_mh_map:
            i = self.ci_mh_map[key_mh]
            return self.ci_mh_low[i], self.ci_mh_high[i]

        return self.global_ci_low, self.global_ci_high

    def compute_ci_features(self, img_raw, idx=None):
        'Data processing note.'
        ci_band = max(0, min(self.ci_band_index, img_raw.shape[0] - 1))
        rho = img_raw[ci_band].astype(np.float32)

        if self.ci_background_mode == "latband_month_localhour" and idx is not None:
            rho_clr, rho_cld = self.get_ci_background_for_sample(idx)
            rho_clr = np.asarray(rho_clr, dtype=np.float32)
            rho_cld = np.asarray(rho_cld, dtype=np.float32)
        else:
            rho_clr = float(self.ci_clear)
            rho_cld = float(self.ci_cloud)

        denom = rho_cld - rho_clr
        ci = (rho - rho_clr) / (denom + 1e-6)
        ci = np.nan_to_num(ci, nan=0.0, posinf=1.0, neginf=0.0)
        ci = np.clip(ci, 0.0, 1.0)

        ci_mean = float(np.mean(ci))
        ci_high = float(np.mean(ci > CONFIG["CI_HIGH_THRESHOLD"]))
        ci_mid = float(np.mean(4.0 * ci * (1.0 - ci)))  #
        gx = np.abs(np.diff(ci, axis=1)).mean() if ci.shape[1] > 1 else 0.0
        gy = np.abs(np.diff(ci, axis=0)).mean() if ci.shape[0] > 1 else 0.0
        ci_edge = float(gx + gy)

        return np.array([ci_mean, ci_high, ci_mid, ci_edge], dtype=np.float32)

    def compute_physical_features(self, raw_ang, doy, ci_features):
        'Data processing note.'
        soz_rad = np.deg2rad(float(raw_ang[2]))
        mu0 = float(np.cos(soz_rad))

        dr = float(earth_sun_distance_factor(np.array([doy], dtype=np.float32))[0])
        # Manuscript Eq. (5): extraterrestrial NORMAL irradiance, without mu0.
        # Horizontal projection is a different quantity, used only for diagnostics.
        rtoa = CONFIG["SOLAR_CONSTANT"] * dr
        rtoa_norm = rtoa / CONFIG["RTOA_SCALE"]

        # Manuscript Eq. (7): Kasten-Young (1989), without a daytime cap.
        # The value for night-time rows is only a numerical fallback; it must
        # not be interpreted as the Kasten-Young formula beyond its domain.
        soz_deg = float(raw_ang[2])
        if 0.0 <= soz_deg < 90.0:
            airmass = 1.0 / (
                np.cos(np.deg2rad(soz_deg))
                + 0.50572 * (96.07995 - soz_deg) ** (-1.6364)
            )
        else:
            airmass = 10.0
        airmass_norm = airmass / 10.0
        ci_mean, ci_high, ci_mid, ci_edge = ci_features
        phys = np.array([
            rtoa_norm,
            mu0,
            airmass_norm,
            ci_mean,
            ci_high,
            ci_mid,
            ci_edge,
        ], dtype=np.float32)
        return phys

    def _ensure_feature_cache(self):
        cache_paths = get_feature_cache_paths(CONFIG["FEATURE_CACHE_PREFIX"])
        os.makedirs(os.path.dirname(cache_paths["meta"]), exist_ok=True)

        expected_meta = {
            "n": int(len(self.images)),
            "feature_schema_version": PREPROCESSING_SCHEMA,
            "data_signature": {key: file_signature(CONFIG[key]) for key in ("IMG_PATH", "ANG_PATH", "LBL_PATH")},
            "preprocessing_artifact_hashes": preprocessing_artifact_hashes(),
            "angle_encoding": "cos_sza_cos_vza_cos_raa_v1",
            "airmass_formula": "kasten_young_1989_uncapped_daytime",
            "rtoa_formula": "solar_constant_times_dr_normal_irradiance",
            "solar_constant": float(CONFIG["SOLAR_CONSTANT"]),
            "rtoa_scale": float(CONFIG["RTOA_SCALE"]),
            "ci_band_index": int(self.ci_band_index),
            "ci_background_mode": str(self.ci_background_mode),
            "ci_low_percentile": float(CONFIG["CI_LOW_PERCENTILE"]),
            "ci_high_percentile": float(CONFIG["CI_HIGH_PERCENTILE"]),
            "ci_high_threshold": float(CONFIG["CI_HIGH_THRESHOLD"]),
            "ci_lat_band_width_deg": float(CONFIG["CI_LAT_BAND_WIDTH_DEG"]),
            "ci_local_hour_bin": int(CONFIG["CI_LOCAL_HOUR_BIN"]),
            "train_index_fingerprint": index_fingerprint(
                load_index_file(CONFIG["TRAIN_INDEX_PATH"])
            ),
            "image_file_size": int(os.path.getsize(clean_path(CONFIG["IMG_PATH"]))),
        }

        cache_ok = all(os.path.exists(cache_paths[k]) for k in ["ang", "phys", "y", "meta"])
        if cache_ok:
            try:
                with open(cache_paths["meta"], "r") as f:
                    old_meta = json.load(f)
                if all(old_meta.get(k) == v for k, v in expected_meta.items()):
                    print('Data processing note.')
                    return
                print('Data processing note.')
            except Exception:
                print('Data processing note.')

        n = len(self.images)
        print('Data processing note.')
        ang_all = np.empty((n, 3), dtype=np.float32)
        phys_all = np.empty((n, 7), dtype=np.float32)
        y_all = np.empty((n, 3), dtype=np.float32)

        for i in tqdm(range(n), desc="Precompute CPG features"):
            img = np.array(self.images[i], dtype=np.float32)
            if not np.isfinite(img).all():
                raise ValueError("Invalid reflectance in sample {}; apply upstream quality control".format(i))
            raw_ang = np.array(self.angles[i], dtype=np.float32)
            ci_features = self.compute_ci_features(img, idx=i)
            ang_all[i] = self.process_angles(raw_ang)
            phys_all[i] = self.compute_physical_features(raw_ang, self.doy[i], ci_features)
            y_all[i] = (self.labels[i] - self.lbl_mean) / self.lbl_std

        np.save(cache_paths["ang"], ang_all)
        np.save(cache_paths["phys"], phys_all)
        np.save(cache_paths["y"], y_all)
        with open(cache_paths["meta"], "w") as f:
            json.dump(expected_meta, f, indent=2)
        print('Data processing note.')

    def __getitem__(self, idx):
        img = np.array(self.images[idx], dtype=np.float32)  # (6, 16, 16)
        if not np.isfinite(img).all():
            raise ValueError("Invalid reflectance in sample {}; apply upstream quality control".format(idx))
        img_norm = (img - self.img_mean) / self.img_std

        if self.use_feature_cache and self.ang_cache is not None:
            ang_encoded = np.array(self.ang_cache[idx], dtype=np.float32)
            phys = np.array(self.phys_cache[idx], dtype=np.float32)
            label_norm = np.array(self.y_cache[idx], dtype=np.float32)
        else:
            ci_features = self.compute_ci_features(img, idx=idx)
            raw_ang = np.array(self.angles[idx]).astype(np.float32)
            ang_encoded = self.process_angles(raw_ang)
            phys = self.compute_physical_features(raw_ang, self.doy[idx], ci_features)
            label_norm = ((self.labels[idx] - self.lbl_mean) / self.lbl_std
                          if self.labels is not None else None)

        result = {
            "x_img": torch.from_numpy(img_norm),
            "x_ang": torch.from_numpy(ang_encoded),
            "x_phys": torch.from_numpy(phys),
            "soz": torch.tensor(self.soz[idx]),
        }
        if label_norm is not None:
            result["y"] = torch.from_numpy(label_norm)
        return result


class _DenseLayer(nn.Module):
    def __init__(self, in_c, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, bn_size * growth_rate, 1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_c, bn_size, growth_rate, drop_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(_DenseLayer(in_c + i * growth_rate, growth_rate, bn_size, drop_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, 1)


class _Transition(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))


class PhysicalGuidanceGate(nn.Module):
    'Data processing note.'
    def __init__(self, phys_dim, feature_dim, gate_hidden=64, gamma=0.30):
        super().__init__()
        self.gamma = float(gamma)
        self.mlp = nn.Sequential(
            nn.Linear(phys_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, feature_dim),
            nn.Sigmoid(),
        )
        self.mlp.apply(initialize_linear_layers)

    def forward(self, h, p):
        gate = self.mlp(p)  # [B, feature_dim], 0-1
        scale = 1.0 + self.gamma * (2.0 * gate - 1.0)
        return h * scale, gate


class CPGDeepDenseNet16(nn.Module):
    'Data processing note.'
    def __init__(self, growth_rate=16, block_config=(4, 6, 4), num_init_features=32,
                 bn_size=4, drop_rate=0.2, gate_gamma=0.30):
        super().__init__()

        # Backbone: Input is already 16x16, no upsampling needed
        self.features = nn.Sequential()
        self.features.add_module("conv0", nn.Conv2d(6, num_init_features, 3, padding=1, bias=False))
        self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
        self.features.add_module("relu0", nn.ReLU(inplace=True))

        num_features = num_init_features
        # 16x16 -> Block1 -> Trans(8x8) -> Block2 -> Trans(4x4) -> Block3 -> GAP
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.img_dim = num_features
        self.features.apply(initialize_convolution_layers)

        # Geometry branch: cosine encoding
        # [cos(SZA), cos(VZA), cos(RAA)].
        self.ang_fc = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True)
        )

        # Fusion shared backbone
        self.shared = nn.Sequential(
            nn.Linear(self.img_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )

        # Apply the physical gate directly to the 128-D shared
        # representation; no raw-angle shortcut is appended here.
        self.feature_dim = 128

        # Component-specific physical guidance gates
        # Rg: normalized [Rtoa, mu0, airmass, CI_mean]; Rtoa is normal irradiance.
        self.gate_g = PhysicalGuidanceGate(phys_dim=4, feature_dim=self.feature_dim, gamma=gate_gamma)
        # Rdir: normalized [Rtoa, mu0, airmass, CI_mean, CI_high]
        self.gate_dir = PhysicalGuidanceGate(phys_dim=5, feature_dim=self.feature_dim, gamma=gate_gamma)
        # Rdif: normalized [Rtoa, mu0, airmass, CI_mean, CI_high, CI_mid, CI_edge]
        self.gate_dif = PhysicalGuidanceGate(phys_dim=7, feature_dim=self.feature_dim, gamma=gate_gamma)
        self.head_g = nn.Sequential(
            nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )
        self.head_dir = nn.Sequential(
            nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )
        self.head_dif = nn.Sequential(
            nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )
        self.ang_fc.apply(initialize_linear_layers)
        self.shared.apply(initialize_linear_layers)
        self.head_g.apply(initialize_linear_layers)
        self.head_dir.apply(initialize_linear_layers)
        self.head_dif.apply(initialize_linear_layers)

    def forward(self, img, ang, phys):
        # img: [B, 6, 16, 16]
        f = self.features(img)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        img_feat = torch.flatten(f, 1)

        ang_feat = self.ang_fc(ang)
        combined = torch.cat((img_feat, ang_feat), dim=1)
        shared_feat = self.shared(combined)

        final_feat = shared_feat  # [B, 128]

        # phys columns
        rtoa_norm = phys[:, 0:1]
        mu0 = phys[:, 1:2]
        airmass_norm = phys[:, 2:3]
        ci_mean = phys[:, 3:4]
        ci_high = phys[:, 4:5]
        ci_mid = phys[:, 5:6]
        ci_edge = phys[:, 6:7]

        # branch-specific physical descriptors
        p_g = torch.cat([rtoa_norm, mu0, airmass_norm, ci_mean], dim=1)
        p_dir = torch.cat([
            rtoa_norm, mu0, airmass_norm, ci_mean, ci_high,
        ], dim=1)
        p_dif = torch.cat([
            rtoa_norm, mu0, airmass_norm, ci_mean,
            ci_high, ci_mid, ci_edge,
        ], dim=1)
        h_g, gate_g = self.gate_g(final_feat, p_g)
        h_dir, gate_dir = self.gate_dir(final_feat, p_dir)
        h_dif, gate_dif = self.gate_dif(final_feat, p_dif)

        # Each task-specific branch directly maps its
        # guided representation to the corresponding radiation estimate.
        pred_g = self.head_g(h_g)
        pred_dir = self.head_dir(h_dir)
        pred_dif = self.head_dif(h_dif)

        # Beam transmittance is retained only as an evaluation diagnostic; it is
        # not an output parameterization and does not affect the forward estimates.
        rtoa = torch.clamp(rtoa_norm * CONFIG["RTOA_SCALE"], min=0.0)
        rtoa_h = rtoa * torch.clamp(mu0, min=0.0)
        pred_tau = torch.clamp(
            pred_dir / torch.clamp(rtoa_h, min=1.0),
            min=0.0,
            max=float(CONFIG["DIRECT_TAU_MAX"]),
        )

        aux = {
            "gate_g_mean": gate_g.mean().detach(),
            "gate_dir_mean": gate_dir.mean().detach(),
            "gate_dif_mean": gate_dif.mean().detach(),
            "pred_tau": pred_tau,
            "rtoa": rtoa,
            "rtoa_h": rtoa_h,
        }
        return pred_g, pred_dir, pred_dif, aux


class AdaptiveMSESoftClosureLoss(nn.Module):
    """Uncertainty-weighted supervised MSEs plus normalized soft closure.

    The three component targets are standardized with training-only statistics.
    The closure residual is divided by the training-set GHI standard deviation,
    using the GHI standard deviation. All four losses have learnable
    homoscedastic uncertainty weights.
    """

    def __init__(self, stats_path, log_vars_init):
        super().__init__()
        if len(log_vars_init) != 4:
            raise ValueError("ADAPTIVE_LOG_VARS_INIT must contain four values")
        self.log_vars = nn.Parameter(torch.tensor(log_vars_init, dtype=torch.float32))
        with open(stats_path, "r") as handle:
            stats = json.load(handle)
        self.register_buffer("lbl_mean", torch.tensor(stats["lbl_mean"], dtype=torch.float32))
        self.register_buffer("lbl_std", torch.tensor(stats["lbl_std"], dtype=torch.float32))

    def forward(self, p_g, p_dir, p_dif, t_g, t_dir, t_dif):
        # Component supervision: all three terms are MSE in standardized space.
        p_g_z = (p_g - self.lbl_mean[0]) / self.lbl_std[0]
        p_dir_z = (p_dir - self.lbl_mean[1]) / self.lbl_std[1]
        p_dif_z = (p_dif - self.lbl_mean[2]) / self.lbl_std[2]
        l_g = F.mse_loss(p_g_z, t_g, reduction="mean")
        l_dir = F.mse_loss(p_dir_z, t_dir, reduction="mean")
        l_dif = F.mse_loss(p_dif_z, t_dif, reduction="mean")

        # Normalize the physical closure residual by the training GHI standard deviation.
        closure_residual = (p_g - (p_dir + p_dif)) / self.lbl_std[0]
        l_closure = torch.mean(torch.square(closure_residual))

        # Uncertainty weighting of all four losses.
        task_losses = torch.stack([l_g, l_dir, l_dif, l_closure])
        precision = torch.exp(-self.log_vars)
        adaptive = torch.sum(0.5 * precision * task_losses + 0.5 * self.log_vars)
        loss = adaptive

        # Fixed monitor is used only for comparable epoch selection.
        monitor = torch.sum(task_losses)
        logs = {
            "loss_g": float(l_g.detach().cpu()),
            "loss_dir": float(l_dir.detach().cpu()),
            "loss_dif": float(l_dif.detach().cpu()),
            "loss_phy": float(l_closure.detach().cpu()),
            "monitor": float(monitor.detach().cpu()),
            "weight_g": float((0.5 * precision[0]).detach().cpu()),
            "weight_dir": float((0.5 * precision[1]).detach().cpu()),
            "weight_dif": float((0.5 * precision[2]).detach().cpu()),
            "weight_phy": float((0.5 * precision[3]).detach().cpu()),
        }
        return loss, logs



def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_model():
    return CPGDeepDenseNet16(
        growth_rate=16,
        block_config=(4, 6, 4),
        num_init_features=32,
        gate_gamma=CONFIG["GATE_GAMMA"],
    )
