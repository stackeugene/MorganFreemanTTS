#!/bin/bash

# Function to add rules
add_rule() {
  echo "Adding rule for $1..."
  sudo ufw allow $1
}

# Function to remove rules
remove_rule() {
  echo "Removing rule for $1..."
  sudo ufw delete allow $1
}

# Step 1: Set default policies
echo "Setting default policies..."
sudo ufw default deny incoming  # Block all incoming traffic by default
sudo ufw default allow outgoing  # Allow all outgoing traffic

# Step 2: Allow essential ports for the TTS program
# Secure SSH Access (Ensure proper syntax)
add_rule "22/tcp"  # Allow SSH (OpenSSH)

# API Port (5000) - Change this if your TTS program runs on a different port
add_rule "5000/tcp"

# HTTPS (for secure web access if needed)
add_rule "443/tcp"

# Optional: Restrict API Access to a Specific IP (Replace YOUR_IP_ADDRESS)
# echo "Restricting API access to specific IP..."
# sudo ufw allow from YOUR_IP_ADDRESS to any port 5000

# Step 3: Enable Rate Limiting to Prevent Abuse
echo "Enabling rate limiting..."
sudo ufw limit 22/tcp  # Protect SSH from brute force attacks
sudo ufw limit 5000/tcp  # Limit API request rate to prevent abuse

# Step 4: Block Unused and Risky Ports (Optional Hardening)
echo "Blocking unnecessary ports..."
sudo ufw deny 80/tcp  # Block HTTP to force HTTPS usage
sudo ufw deny 23/tcp  # Block Telnet (insecure)
sudo ufw deny 3389/tcp  # Block RDP (if not used)

# Step 5: Enable UFW Logging for Monitoring
echo "Enabling firewall logging..."
sudo ufw logging on

# Step 6: Enable the firewall (if not already enabled)
echo "Enabling the firewall..."
sudo ufw enable

# Step 7: Print the current status of the firewall for review
echo "Current firewall status:"
sudo ufw status verbose

# Optional: Reset firewall rules (commented out for safety)
# Uncomment the following lines if you need to reset all rules
# echo "Resetting firewall rules..."
# sudo ufw reset
