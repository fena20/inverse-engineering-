"""
Data loader module for multi-city EPC data.

This module handles loading and merging EPC data from multiple UK cities,
including certificates, recommendations, and weather data.
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..utils.constants import (
    LA_CODE_TO_CITY,
    CITY_CLIMATE_DATA
)


class DataLoader:
    """
    Loads and merges EPC data from multiple cities.
    
    Attributes:
        data_dir: Path to the data directory
        cities: List of cities to load
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory containing city folders
        """
        self.data_dir = Path(data_dir)
        self.cities = []
        self._certificates_df = None
        self._recommendations_df = None
        self._weather_data = {}
        
    def discover_cities(self) -> List[str]:
        """
        Discover available city datasets in the data directory.
        
        Returns:
            List of city names found
        """
        self.cities = []
        city_dirs = glob.glob(str(self.data_dir / "domestic-*"))
        
        for city_dir in city_dirs:
            dir_name = os.path.basename(city_dir)
            # Extract LA code from directory name
            parts = dir_name.split('-')
            if len(parts) >= 2:
                la_code = parts[1]
                if la_code in LA_CODE_TO_CITY:
                    self.cities.append(LA_CODE_TO_CITY[la_code])
        
        return self.cities
    
    def load_certificates(self, cities: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load certificate data from specified cities.
        
        Args:
            cities: List of city names to load. If None, loads all discovered cities.
        
        Returns:
            Combined DataFrame of all certificates
        """
        if cities is None:
            if not self.cities:
                self.discover_cities()
            cities = self.cities
        
        dfs = []
        
        for city in cities:
            city_data = CITY_CLIMATE_DATA.get(city)
            if city_data is None:
                print(f"Warning: City '{city}' not found in climate data")
                continue
            
            la_code = city_data['code']
            
            # Find the directory for this city
            city_dirs = glob.glob(str(self.data_dir / f"domestic-{la_code}-*"))
            if not city_dirs:
                print(f"Warning: No data directory found for {city}")
                continue
            
            cert_file = Path(city_dirs[0]) / "certificates.csv"
            if not cert_file.exists():
                print(f"Warning: No certificates file found for {city}")
                continue
            
            print(f"Loading {city} certificates...")
            df = pd.read_csv(cert_file, low_memory=False)
            df['CITY'] = city
            df['HDD'] = city_data['hdd_base_15_5']
            df['AVG_TEMP'] = city_data['avg_temp']
            df['SOLAR_RADIATION'] = city_data['solar_radiation']
            dfs.append(df)
            print(f"  Loaded {len(df):,} records from {city}")
        
        if not dfs:
            raise ValueError("No certificate data loaded")
        
        self._certificates_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal: {len(self._certificates_df):,} certificates from {len(dfs)} cities")
        
        return self._certificates_df
    
    def load_recommendations(self, cities: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load recommendation data from specified cities.
        
        Args:
            cities: List of city names to load. If None, loads all discovered cities.
        
        Returns:
            Combined DataFrame of all recommendations
        """
        if cities is None:
            if not self.cities:
                self.discover_cities()
            cities = self.cities
        
        dfs = []
        
        for city in cities:
            city_data = CITY_CLIMATE_DATA.get(city)
            if city_data is None:
                continue
            
            la_code = city_data['code']
            city_dirs = glob.glob(str(self.data_dir / f"domestic-{la_code}-*"))
            
            if not city_dirs:
                continue
            
            rec_file = Path(city_dirs[0]) / "recommendations.csv"
            if not rec_file.exists():
                continue
            
            print(f"Loading {city} recommendations...")
            df = pd.read_csv(rec_file, low_memory=False)
            df['CITY'] = city
            dfs.append(df)
            print(f"  Loaded {len(df):,} recommendations from {city}")
        
        if not dfs:
            raise ValueError("No recommendation data loaded")
        
        self._recommendations_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal: {len(self._recommendations_df):,} recommendations")
        
        return self._recommendations_df
    
    def load_weather_data(self, cities: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load weather data from Open-Meteo CSV files.
        
        Args:
            cities: List of city names to load
        
        Returns:
            Dictionary mapping city names to weather DataFrames
        """
        if cities is None:
            if not self.cities:
                self.discover_cities()
            cities = self.cities
        
        self._weather_data = {}
        
        for city in cities:
            city_data = CITY_CLIMATE_DATA.get(city)
            if city_data is None:
                continue
            
            la_code = city_data['code']
            city_dirs = glob.glob(str(self.data_dir / f"domestic-{la_code}-*"))
            
            if not city_dirs:
                continue
            
            # Find weather file (open-meteo-*.csv)
            weather_files = glob.glob(str(Path(city_dirs[0]) / "open-meteo-*.csv"))
            
            if not weather_files:
                continue
            
            print(f"Loading {city} weather data...")
            # Read with multi-header
            df = pd.read_csv(weather_files[0], skiprows=2, low_memory=False)
            self._weather_data[city] = df
            print(f"  Loaded {len(df):,} weather records from {city}")
        
        return self._weather_data
    
    def get_postcode_groups(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Extract postcode sectors for geographic grouping.
        
        This is used to ensure train/test split doesn't have data leakage
        from the same postcode areas.
        
        Args:
            df: DataFrame with POSTCODE column
        
        Returns:
            Series of postcode sectors (e.g., 'CB1 3')
        """
        if df is None:
            df = self._certificates_df
        
        if df is None:
            raise ValueError("No data loaded. Call load_certificates first.")
        
        # Extract postcode sector (e.g., CB1 3 from CB1 3TN)
        def get_sector(postcode):
            if pd.isna(postcode):
                return 'UNKNOWN'
            parts = str(postcode).split(' ')
            if len(parts) >= 2:
                # Return outward code + first digit of inward
                return parts[0] + ' ' + parts[1][0] if len(parts[1]) > 0 else parts[0]
            return parts[0] if parts else 'UNKNOWN'
        
        return df['POSTCODE'].apply(get_sector)
    
    def get_merged_data(
        self,
        cities: Optional[List[str]] = None,
        include_recommendations: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Get merged certificates and recommendations data.
        
        Args:
            cities: List of cities to load
            include_recommendations: Whether to include recommendations
        
        Returns:
            Tuple of (certificates_df, recommendations_df)
        """
        certs = self.load_certificates(cities)
        
        if include_recommendations:
            recs = self.load_recommendations(cities)
            return certs, recs
        
        return certs, None
    
    @property
    def certificates(self) -> Optional[pd.DataFrame]:
        """Get loaded certificates DataFrame."""
        return self._certificates_df
    
    @property
    def recommendations(self) -> Optional[pd.DataFrame]:
        """Get loaded recommendations DataFrame."""
        return self._recommendations_df
    
    @property
    def weather(self) -> Dict[str, pd.DataFrame]:
        """Get loaded weather data."""
        return self._weather_data


def load_all_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all available data.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Tuple of (certificates_df, recommendations_df)
    """
    loader = DataLoader(data_dir)
    loader.discover_cities()
    return loader.get_merged_data()
