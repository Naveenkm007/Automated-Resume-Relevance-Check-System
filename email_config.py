#!/usr/bin/env python3
"""
Email Configuration Setup
========================

Easy configuration setup for email automation.
Built for Innomatics Research Labs.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmailSettings:
    """Email settings configuration."""
    
    # SMTP Configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    
    # Sender Configuration  
    sender_email: str = ""
    sender_password: str = ""  # Use App Password for Gmail
    sender_name: str = "Innomatics Research Labs - HR Team"
    
    # Recipients
    hr_email: str = ""
    cc_emails: list = None
    
    # Automation Settings
    score_threshold: float = 90.0
    auto_send_enabled: bool = True
    
    def __post_init__(self):
        if self.cc_emails is None:
            self.cc_emails = []

def setup_email_config() -> EmailSettings:
    """
    Setup email configuration with user input or environment variables.
    
    Priority:
    1. Environment variables
    2. User input prompts
    3. Default values
    """
    
    print("🔧 Email Configuration Setup - Innomatics Research Labs")
    print("=" * 60)
    
    # Try to load from environment first
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD") 
    hr_email = os.getenv("HR_EMAIL")
    
    # If not found in environment, prompt user
    if not sender_email:
        print("\n📧 Sender Email Configuration:")
        sender_email = input("Enter sender email (Gmail recommended): ").strip()
        
    if not sender_password:
        print("\n🔑 Email Password Configuration:")
        print("For Gmail, use App Password (not regular password)")
        print("Setup: Gmail → Security → 2FA → App Passwords")
        sender_password = input("Enter email app password: ").strip()
        
    if not hr_email:
        print("\n👥 HR Team Configuration:")
        hr_email = input("Enter HR team email: ").strip()
    
    # Score threshold
    print(f"\n🎯 Score Threshold (default: 90%):")
    threshold_input = input("Enter threshold (press Enter for 90%): ").strip()
    score_threshold = float(threshold_input) if threshold_input else 90.0
    
    # Auto-send confirmation
    print(f"\n🤖 Auto-send emails for scores ≥ {score_threshold}%?")
    auto_send = input("Enable auto-send? (Y/n): ").strip().lower()
    auto_send_enabled = auto_send != 'n'
    
    config = EmailSettings(
        sender_email=sender_email,
        sender_password=sender_password,
        hr_email=hr_email,
        score_threshold=score_threshold,
        auto_send_enabled=auto_send_enabled
    )
    
    print("\n✅ Email configuration completed!")
    print(f"   Sender: {config.sender_email}")
    print(f"   HR Email: {config.hr_email}")
    print(f"   Threshold: {config.score_threshold}%")
    print(f"   Auto-send: {'Enabled' if config.auto_send_enabled else 'Disabled'}")
    
    # Save to environment file for future use
    save_config_to_env(config)
    
    return config

def save_config_to_env(config: EmailSettings):
    """Save configuration to .env file."""
    try:
        with open(".env", "w") as f:
            f.write(f"# Innomatics Research Labs - Email Configuration\n")
            f.write(f"SENDER_EMAIL={config.sender_email}\n")
            f.write(f"SENDER_PASSWORD={config.sender_password}\n")  
            f.write(f"HR_EMAIL={config.hr_email}\n")
            f.write(f"SCORE_THRESHOLD={config.score_threshold}\n")
            f.write(f"AUTO_SEND_ENABLED={config.auto_send_enabled}\n")
            
        print("💾 Configuration saved to .env file")
        
    except Exception as e:
        print(f"⚠️ Could not save to .env file: {e}")

def load_config_from_env() -> Optional[EmailSettings]:
    """Load configuration from .env file."""
    try:
        if not os.path.exists(".env"):
            return None
            
        config_dict = {}
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    config_dict[key] = value
        
        return EmailSettings(
            sender_email=config_dict.get("SENDER_EMAIL", ""),
            sender_password=config_dict.get("SENDER_PASSWORD", ""),
            hr_email=config_dict.get("HR_EMAIL", ""),
            score_threshold=float(config_dict.get("SCORE_THRESHOLD", 90.0)),
            auto_send_enabled=config_dict.get("AUTO_SEND_ENABLED", "True").lower() == "true"
        )
        
    except Exception as e:
        print(f"⚠️ Error loading config from .env: {e}")
        return None

if __name__ == "__main__":
    # Interactive setup
    config = setup_email_config()
    
    print("\n🧪 Test Configuration:")
    test = input("Send test email? (y/N): ").strip().lower()
    
    if test == 'y':
        from email_automation import EmailAutomation, EmailConfig, CandidateData
        
        # Convert to EmailConfig
        email_config = EmailConfig(
            sender_email=config.sender_email,
            sender_password=config.sender_password,
            hr_email=config.hr_email,
            sender_name=config.sender_name
        )
        
        # Test candidate
        test_candidate = CandidateData(
            name="Test Candidate",
            email=config.sender_email,  # Send to yourself for testing
            phone="+91-9876543210", 
            overall_score=92.0,
            job_title="Test Position",
            matched_skills=["python", "testing"],
            missing_skills=[],
            resume_text="This is a test..."
        )
        
        # Send test emails
        email_system = EmailAutomation(email_config)
        results = email_system.process_candidate_score(test_candidate, config.score_threshold)
        
        print("\n📧 Test Results:")
        print(f"   Candidate email: {'✅ Sent' if results['candidate_email_sent'] else '❌ Failed'}")
        print(f"   HR notification: {'✅ Sent' if results['hr_notification_sent'] else '❌ Failed'}")
