#!/usr/bin/env python3
"""
Email Automation System for Resume Relevance Check
=================================================

Automatically send emails to high-scoring candidates (90%+) and HR notifications.
Built for Innomatics Research Labs.

Features:
- Automated email sending for candidates scoring 90%+
- Customizable email templates
- HR notification system
- Email tracking and logging
- SMTP configuration support
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration settings."""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    sender_name: str = "Innomatics Research Labs"
    hr_email: str = ""

@dataclass 
class CandidateData:
    """Candidate information for email processing."""
    name: str
    email: str
    phone: str
    overall_score: float
    job_title: str
    matched_skills: List[str]
    missing_skills: List[str]
    resume_text: str

class EmailAutomation:
    """Main email automation class."""
    
    def __init__(self, config: EmailConfig):
        self.config = config
        self.email_log = []
        
    def setup_smtp_connection(self):
        """Setup and return SMTP connection."""
        try:
            # Create SMTP session
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()  # Enable TLS encryption
            server.login(self.config.sender_email, self.config.sender_password)
            return server
        except Exception as e:
            logger.error(f"Failed to setup SMTP connection: {e}")
            raise
    
    def generate_candidate_email_template(self, candidate: CandidateData) -> str:
        """Generate personalized email template for high-scoring candidates."""
        
        template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; background: #f9f9f9; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; text-align: center; }}
                .content {{ padding: 2rem; background: white; }}
                .score-highlight {{ background: #28a745; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; display: inline-block; }}
                .skills {{ background: #f8f9fa; padding: 1rem; border-left: 4px solid #667eea; margin: 1rem 0; }}
                .footer {{ background: #333; color: white; padding: 1rem; text-align: center; }}
                .btn {{ background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 1rem 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Congratulations, {candidate.name}!</h1>
                    <p>You've scored exceptionally well in our AI-powered resume analysis</p>
                </div>
                
                <div class="content">
                    <p>Dear {candidate.name},</p>
                    
                    <p>We are excited to inform you that your profile has achieved an outstanding score in our automated resume relevance analysis for the <strong>{candidate.job_title}</strong> position.</p>
                    
                    <div style="text-align: center; margin: 2rem 0;">
                        <div class="score-highlight">
                            🏆 Your Score: {candidate.overall_score:.1f}%
                        </div>
                        <p><em>Excellent Match - Top Candidate!</em></p>
                    </div>
                    
                    <h3>🎯 Your Matched Skills:</h3>
                    <div class="skills">
                        {', '.join([skill.title() for skill in candidate.matched_skills[:8]])}
                        {f"<br><small>...and {len(candidate.matched_skills) - 8} more!</small>" if len(candidate.matched_skills) > 8 else ""}
                    </div>
                    
                    <p><strong>Next Steps:</strong></p>
                    <ul>
                        <li>📞 Our HR team will contact you within 2 business days</li>
                        <li>📋 Prepare for a technical interview discussion</li>
                        <li>💼 Review the complete job description attached</li>
                    </ul>
                    
                    <div style="text-align: center;">
                        <a href="mailto:{self.config.sender_email}" class="btn">Contact HR Team</a>
                    </div>
                    
                    <p>We look forward to discussing this exciting opportunity with you!</p>
                    
                    <p>Best regards,<br>
                    <strong>HR Team</strong><br>
                    Innomatics Research Labs</p>
                </div>
                
                <div class="footer">
                    <p>🤖 This email was generated by our AI-powered recruitment system</p>
                    <p>📧 {self.config.sender_email} | 📱 Contact: +91-XXXX-XXXX</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return template
    
    def generate_hr_notification_template(self, candidate: CandidateData) -> str:
        """Generate HR notification email for high-scoring candidates."""
        
        template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 700px; margin: 0 auto; }}
                .header {{ background: #dc3545; color: white; padding: 1rem; text-align: center; }}
                .content {{ padding: 1.5rem; background: #f8f9fa; }}
                .candidate-info {{ background: white; padding: 1rem; border-left: 4px solid #28a745; margin: 1rem 0; }}
                .score {{ font-size: 1.5rem; font-weight: bold; color: #28a745; }}
                .skills-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }}
                .urgent {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🚨 HIGH-PRIORITY CANDIDATE ALERT</h2>
                    <p>Automated Resume Analysis - Action Required</p>
                </div>
                
                <div class="content">
                    <div class="urgent">
                        <h3>⚡ IMMEDIATE ACTION REQUIRED</h3>
                        <p>A candidate has scored <strong>{candidate.overall_score:.1f}%</strong> - exceeding the 90% threshold for automatic contact.</p>
                    </div>
                    
                    <div class="candidate-info">
                        <h3>👤 Candidate Details</h3>
                        <p><strong>Name:</strong> {candidate.name}</p>
                        <p><strong>Email:</strong> {candidate.email}</p>
                        <p><strong>Phone:</strong> {candidate.phone}</p>
                        <p><strong>Position:</strong> {candidate.job_title}</p>
                        <p><strong>Overall Score:</strong> <span class="score">{candidate.overall_score:.1f}%</span></p>
                    </div>
                    
                    <div class="skills-grid">
                        <div>
                            <h4>✅ Matched Skills ({len(candidate.matched_skills)})</h4>
                            <ul>
                                {chr(10).join([f"<li>{skill.title()}</li>" for skill in candidate.matched_skills[:10]])}
                            </ul>
                        </div>
                        <div>
                            <h4>⚠️ Missing Skills ({len(candidate.missing_skills)})</h4>
                            <ul>
                                {chr(10).join([f"<li>{skill.title()}</li>" for skill in candidate.missing_skills[:5]]) if candidate.missing_skills else "<li>None - Excellent match!</li>"}
                            </ul>
                        </div>
                    </div>
                    
                    <h3>📋 Recommended Next Steps:</h3>
                    <ol>
                        <li><strong>Contact candidate within 24 hours</strong></li>
                        <li>Schedule technical interview</li>
                        <li>Verify skill claims during interview</li>
                        <li>Consider fast-track hiring process</li>
                    </ol>
                    
                    <div style="background: #e7f3ff; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                        <h4>🤖 AI Analysis Summary:</h4>
                        <p>This candidate demonstrates exceptional alignment with job requirements. The AI system recommends immediate engagement to prevent loss to competitors.</p>
                    </div>
                </div>
                
                <div style="background: #333; color: white; padding: 1rem; text-align: center;">
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Innomatics Research Labs - AI Recruitment System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return template
    
    def send_candidate_email(self, candidate: CandidateData) -> bool:
        """Send congratulations email to high-scoring candidate."""
        try:
            if not candidate.email:
                logger.warning(f"No email found for candidate {candidate.name}")
                return False
            
            # Generate email content
            html_content = self.generate_candidate_email_template(candidate)
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"🎉 Exciting Opportunity - {candidate.job_title} Position | Innomatics Research Labs"
            message["From"] = f"{self.config.sender_name} <{self.config.sender_email}>"
            message["To"] = candidate.email
            
            # Add HTML content
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            server = self.setup_smtp_connection()
            server.send_message(message)
            server.quit()
            
            # Log successful send
            self.log_email_sent("candidate", candidate.name, candidate.email, candidate.overall_score)
            logger.info(f"Successfully sent candidate email to {candidate.name} ({candidate.email})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send candidate email to {candidate.name}: {e}")
            return False
    
    def send_hr_notification(self, candidate: CandidateData) -> bool:
        """Send notification to HR team about high-scoring candidate."""
        try:
            if not self.config.hr_email:
                logger.warning("No HR email configured")
                return False
            
            # Generate email content
            html_content = self.generate_hr_notification_template(candidate)
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"🚨 HIGH-PRIORITY: {candidate.name} scored {candidate.overall_score:.1f}% - Immediate Action Required"
            message["From"] = f"AI Recruitment System <{self.config.sender_email}>"
            message["To"] = self.config.hr_email
            message["Priority"] = "1"  # High priority
            
            # Add HTML content
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            server = self.setup_smtp_connection()
            server.send_message(message)
            server.quit()
            
            # Log successful send
            self.log_email_sent("hr_notification", candidate.name, self.config.hr_email, candidate.overall_score)
            logger.info(f"Successfully sent HR notification for {candidate.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send HR notification for {candidate.name}: {e}")
            return False
    
    def process_candidate_score(self, candidate: CandidateData, score_threshold: float = 90.0) -> Dict[str, bool]:
        """
        Main function to process candidate score and trigger emails if above threshold.
        
        Args:
            candidate: CandidateData object
            score_threshold: Minimum score to trigger emails (default: 90.0)
            
        Returns:
            Dict with email send results
        """
        results = {
            "candidate_email_sent": False,
            "hr_notification_sent": False,
            "threshold_met": False
        }
        
        if candidate.overall_score >= score_threshold:
            results["threshold_met"] = True
            logger.info(f"🎯 Candidate {candidate.name} scored {candidate.overall_score:.1f}% - Above {score_threshold}% threshold!")
            
            # Send candidate congratulations email
            results["candidate_email_sent"] = self.send_candidate_email(candidate)
            
            # Send HR notification
            results["hr_notification_sent"] = self.send_hr_notification(candidate)
            
            if results["candidate_email_sent"] and results["hr_notification_sent"]:
                logger.info(f"✅ All emails sent successfully for {candidate.name}")
            else:
                logger.warning(f"⚠️ Some emails failed for {candidate.name}")
        else:
            logger.info(f"Candidate {candidate.name} scored {candidate.overall_score:.1f}% - Below {score_threshold}% threshold")
        
        return results
    
    def log_email_sent(self, email_type: str, candidate_name: str, recipient: str, score: float):
        """Log email sending for tracking purposes."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "email_type": email_type,
            "candidate_name": candidate_name,
            "recipient": recipient,
            "score": score
        }
        self.email_log.append(log_entry)
        
        # Save to file for persistence
        try:
            with open("email_log.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to save email log: {e}")

# Email configuration from environment variables or config file
def load_email_config() -> EmailConfig:
    """Load email configuration from environment variables."""
    return EmailConfig(
        smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        sender_email=os.getenv("SENDER_EMAIL", ""),
        sender_password=os.getenv("SENDER_PASSWORD", ""),
        sender_name=os.getenv("SENDER_NAME", "Innomatics Research Labs"),
        hr_email=os.getenv("HR_EMAIL", "")
    )

# Example usage function
def example_usage():
    """Example of how to use the email automation system."""
    
    # Load configuration
    config = load_email_config()
    
    # Create email automation instance
    email_system = EmailAutomation(config)
    
    # Example candidate data
    candidate = CandidateData(
        name="John Doe",
        email="john.doe@example.com", 
        phone="+91-9876543210",
        overall_score=92.5,
        job_title="Senior Python Developer",
        matched_skills=["python", "django", "postgresql", "aws", "docker"],
        missing_skills=["kubernetes"],
        resume_text="Sample resume text..."
    )
    
    # Process candidate (will send emails if score >= 90%)
    results = email_system.process_candidate_score(candidate)
    
    print("Email Results:", results)

if __name__ == "__main__":
    example_usage()
