#!/usr/bin/env python3
"""
Test script to verify Basketball Reference API functionality
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
from datetime import datetime

def test_basketball_reference_connection():
    """Test basic connection to Basketball Reference"""
    try:
        print("🔍 Testing Basketball Reference connection...")
        url = "https://www.basketball-reference.com"
        response = requests.get(url, timeout=10)
        print(f"✅ Connection successful! Status code: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_nba_games_page():
    """Test access to NBA games page"""
    try:
        print("\n🔍 Testing NBA games page access...")
        # Test with 2024 season (most recent complete season)
        url = "https://www.basketball-reference.com/leagues/NBA_2024_games.html"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        print(f"✅ NBA games page accessible! Status code: {response.status_code}")
        
        # Check if we can parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for the schedule table
        table = soup.find("table", {"id": "schedule"})
        if table:
            print("✅ Schedule table found!")
            
            # Count rows to see if we have data
            rows = table.find_all("tr")
            data_rows = [row for row in rows if row.find_all("td")]
            print(f"✅ Found {len(data_rows)} game records")
            
            if len(data_rows) > 0:
                # Test parsing a sample row
                sample_row = data_rows[0]
                cols = sample_row.find_all("td")
                if len(cols) >= 5:
                    date = cols[0].text.strip()
                    away_team = cols[1].text.strip()
                    away_score = cols[2].text.strip()
                    home_team = cols[3].text.strip()
                    home_score = cols[4].text.strip()
                    
                    print(f"✅ Sample game parsed: {away_team} @ {home_team} on {date}")
                    print(f"   Score: {away_team} {away_score} - {home_team} {home_score}")
                    return True
                else:
                    print("⚠️ Unexpected table structure")
                    return False
            else:
                print("⚠️ No game data found in table")
                return False
        else:
            print("❌ Schedule table not found!")
            print("Page title:", soup.find("title").text if soup.find("title") else "No title")
            return False
            
    except Exception as e:
        print(f"❌ NBA games page test failed: {e}")
        return False

def test_scrape_sample_games():
    """Test the actual scraping function with a small sample"""
    try:
        print("\n🔍 Testing game scraping functionality...")
        
        url = "https://www.basketball-reference.com/leagues/NBA_2024_games.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        table = soup.find("table", {"id": "schedule"})
        if not table:
            print("❌ Could not find schedule table")
            return False
            
        games = []
        rows = table.find_all("tr")
        
        # Process first 5 games as test
        count = 0
        for row in rows:
            if count >= 5:
                break
                
            cols = row.find_all("td")
            if cols and len(cols) >= 5:
                try:
                    date = cols[0].text.strip()
                    away_team = cols[1].text.strip()
                    away_score = cols[2].text.strip()
                    home_team = cols[3].text.strip()
                    home_score = cols[4].text.strip()
                    
                    # Skip if scores are empty (future games)
                    if away_score and home_score:
                        games.append([date, away_team, away_score, home_team, home_score, 2024])
                        count += 1
                        print(f"✅ Parsed: {away_team} {away_score} @ {home_team} {home_score} ({date})")
                except Exception as e:
                    print(f"⚠️ Error parsing row: {e}")
                    continue
        
        if games:
            # Test DataFrame creation
            df = pd.DataFrame(games, columns=["Date", "Away Team", "Away Score", "Home Team", "Home Score", "Season"])
            print(f"\n✅ Successfully created DataFrame with {len(df)} games")
            print("Sample data:")
            print(df.head())
            return True
        else:
            print("❌ No games were successfully parsed")
            return False
            
    except Exception as e:
        print(f"❌ Scraping test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏀 Basketball Reference API Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Basic Connection", test_basketball_reference_connection),
        ("NBA Games Page", test_nba_games_page), 
        ("Sample Scraping", test_scrape_sample_games)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("🏀 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Basketball Reference scraping is functional.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
