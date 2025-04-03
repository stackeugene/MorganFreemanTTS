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
# SSH (for remote access)
add_rule "OpenSSH"

# API Port (change 5000 to the port you are using for TTS)
add_rule "5000/tcp"

# HTTPS (for secure web access)
add_rule "443/tcp"

# Step 3: Optional - Restrict access to certain IPs for sensitive services
# Replace YOUR_IP_ADDRESS with the allowed IP address
# Uncomment and replace 'YOUR_IP_ADDRESS' to use
# echo "Restricting API access to specific IP..."
# sudo ufw allow from YOUR_IP_ADDRESS to any port 5000

# Step 4: Enable UFW Logging to track allowed and denied connections
echo "Enabling firewall logging..."
sudo ufw logging on

# Step 5: Enable rate limiting for SSH to prevent brute-force attacks
echo "Enabling rate limiting for SSH..."
sudo ufw limit OpenSSH

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
