#!/usr/bin/env python3
"""
🔐 BACKPACK API CREDENTIALS CONFIGURATION
Set up your API credentials for testing

IMPORTANT: Keep your API keys secure and never commit them to version control!
"""

import os
from typing import Optional

class BackpackTestConfig:
    """Configuration for Backpack API testing."""
    
    def __init__(self):
        # API Credentials - Set these environment variables or modify directly
        self.api_key = os.getenv('BACKPACK_API_KEY', 'your_api_key_here')
        self.api_secret = os.getenv('BACKPACK_API_SECRET', 'your_api_secret_here')
        
        # Test settings
        self.testnet = True  # Set to False for mainnet testing
        self.test_symbols = ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']
        
        # Test timeouts
        self.stream_test_duration = 10  # seconds
        self.connection_timeout = 30  # seconds
        
        # Rate limiting
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def is_configured(self) -> bool:
        """Check if API credentials are properly configured."""
        return (
            self.api_key and 
            self.api_secret and 
            self.api_key != 'your_api_key_here' and
            self.api_secret != 'your_api_secret_here'
        )
    
    def get_credentials(self) -> tuple:
        """Get API credentials tuple."""
        return (self.api_key, self.api_secret)
    
    def print_status(self):
        """Print configuration status."""
        print("🔐 Backpack Test Configuration")
        print("=" * 40)
        print(f"API Key: {'✅ Set' if self.api_key and self.api_key != 'your_api_key_here' else '❌ Not set'}")
        print(f"API Secret: {'✅ Set' if self.api_secret and self.api_secret != 'your_api_secret_here' else '❌ Not set'}")
        print(f"Testnet: {'✅ Enabled' if self.testnet else '❌ Disabled (MAINNET)'}")
        print(f"Test Symbols: {', '.join(self.test_symbols)}")
        print(f"Fully Configured: {'✅ Yes' if self.is_configured() else '❌ No'}")
        print("=" * 40)
        
        if not self.is_configured():
            print("\n⚠️  TO SET UP CREDENTIALS:")
            print("Option 1 - Environment Variables:")
            print("export BACKPACK_API_KEY='your_actual_api_key'")
            print("export BACKPACK_API_SECRET='your_actual_api_secret'")
            print("\nOption 2 - Direct Configuration:")
            print("Edit this file and set api_key and api_secret directly")
            print("\n🔗 Get API keys from: https://backpack.exchange/settings/api")

# Create global config instance
config = BackpackTestConfig()

# Usage example
if __name__ == "__main__":
    config.print_status()
    
    if config.is_configured():
        print("\n✅ Ready to run tests!")
        print("Run: python test_backpack_adapter.py")
    else:
        print("\n❌ Please configure your API credentials first")