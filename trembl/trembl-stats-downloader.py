import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniProtStats:
    BASE_URL = "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases"

    def __init__(self):
        self.data = []

    def generate_urls(self, release: str) -> Tuple[str, str]:
        """Generate URLs for Swiss-Prot and TrEMBL statistics files."""
        base_path = f"{self.BASE_URL}/release-{release}/knowledgebase"
        swiss_prot_url = f"{base_path}/UniProtKB_SwissProt-relstat.html"
        trembl_url = f"{base_path}/UniProtKB_TrEMBL-relstat.html"
        return swiss_prot_url, trembl_url

    def extract_sequence_count(self, content: str) -> Optional[int]:
        """Extract sequence count from the content using regex."""
        pattern = r"contains\s+(\d+(?:,\d+)*)\s+sequence entries"
        match = re.search(pattern, content)
        if match:
            # Remove commas and convert to integer
            return int(match.group(1).replace(',', ''))
        return None

    def fetch_release_stats(self, release: str) -> Optional[Tuple[int, int]]:
        """Fetch and parse statistics for a specific release."""
        swiss_prot_url, trembl_url = self.generate_urls(release)

        try:
            # Fetch Swiss-Prot stats
            swiss_response = requests.get(swiss_prot_url)
            swiss_response.raise_for_status()
            swiss_count = self.extract_sequence_count(swiss_response.text)

            # Fetch TrEMBL stats
            trembl_response = requests.get(trembl_url)
            trembl_response.raise_for_status()
            trembl_count = self.extract_sequence_count(trembl_response.text)

            if swiss_count and trembl_count:
                return swiss_count, trembl_count

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Release {release} not found")
            else:
                logger.error(f"Error fetching release {release}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for release {release}: {e}")

        return None

    def generate_release_dates(self) -> List[str]:
        """Generate release dates from 2016 to 2024_07."""
        releases = []
        for year in range(2016, 2025):
            for month in range(1, 13):
                # Stop at July 2024
                if year == 2024 and month > 7:
                    break
                releases.append(f"{year}_{month:02d}")
        return releases

    def collect_stats(self) -> pd.DataFrame:
        """Collect statistics for all releases."""
        releases = self.generate_release_dates()

        for release in releases:
            logger.info(f"Processing release {release}")
            stats = self.fetch_release_stats(release)

            if stats:
                swiss_count, trembl_count = stats
                total_count = swiss_count + trembl_count
                ratio = swiss_count / total_count * 100

                self.data.append({
                    'release': release,
                    'swiss_prot_count': swiss_count,
                    'trembl_count': trembl_count,
                    'total_count': total_count,
                    'swiss_prot_percentage': ratio
                })

        return pd.DataFrame(self.data)

def main():
    stats = UniProtStats()
    df = stats.collect_stats()

    # Save to CSV
    df.to_csv('uniprot_stats.csv', index=False)

    # Print summary statistics
    print("\nSummary of Swiss-Prot percentage over time:")
    print(df[['release', 'swiss_prot_percentage']].to_string(index=False))

    # Calculate average ratio
    avg_ratio = df['swiss_prot_percentage'].mean()
    print(f"\nAverage Swiss-Prot percentage: {avg_ratio:.2f}%")

    # Find min and max ratios
    min_ratio = df.loc[df['swiss_prot_percentage'].idxmin()]
    max_ratio = df.loc[df['swiss_prot_percentage'].idxmax()]

    print(f"\nLowest Swiss-Prot percentage: {min_ratio['swiss_prot_percentage']:.2f}% (Release {min_ratio['release']})")
    print(f"Highest Swiss-Prot percentage: {max_ratio['swiss_prot_percentage']:.2f}% (Release {max_ratio['release']})")

if __name__ == "__main__":
    main()