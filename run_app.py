#!/usr/bin/env python3
"""
Script to run the Gen AI Interview Assistant Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit application"""
    print("🚀 Starting Gen AI Interview Assistant...")
    print("=" * 50)

    # Check if we're in the conda environment
    if "genai-interview" not in sys.executable:
        print("⚠️  Warning: Not running in the 'genai-interview' conda environment")
        print("   Run: conda activate genai-interview")
        print()

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if has_openai or has_anthropic:
        print("✅ API keys detected - Full AI functionality enabled")
    else:
        print("ℹ️  No API keys found - Running in retrieval-only mode")
        print("   Add keys for full AI responses:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
    print()

    # Run Streamlit
    app_path = Path(__file__).parent / "src" / "app.py"

    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()