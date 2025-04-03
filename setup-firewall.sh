
# Update system and install UFW
echo "Updating system and installing UFW..."
sudo apt update && sudo apt install ufw -y  # For Ubuntu/Debian-based systems
# If using Amazon Linux or CentOS, you can replace with: sudo yum install ufw -y

# Set default policies
echo "Setting default policies..."
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow OpenSSH to ensure SSH connections remain open
echo "Allowing SSH connections..."
sudo ufw all#!/bin/bash
ow OpenSSH

# Allow port for the TTS program (adjust if you're using a different port)
echo "Allowing port 5000 for TTS API..."
sudo ufw allow 5000/tcp  # Change 5000 to the port you're using if different

# Allow HTTPS traffic if using a web frontend
echo "Allowing HTTPS traffic on port 443..."
sudo ufw allow 443/tcp

# Optional: Restrict access to specific IP address for the API (replace YOUR_IP_ADDRESS)
# echo "Allowing traffic from specific IP address..."
# sudo ufw allow from YOUR_IP_ADDRESS to any port 5000

# Enable rate limiting for SSH to prevent brute-force attacks
echo "Enabling rate limiting for SSH..."
sudo ufw limit OpenSSH

# Enable UFW logging to track attempts
echo "Enabling UFW logging..."
sudo ufw logging on

# Enable the firewall
echo "Enabling the firewall..."
sudo ufw enable

# Display the current status of the firewall
echo "Current firewall status:"
sudo ufw status verbose

# You can reset firewall rules if something goes wrong (commented out, for safety)
# echo "Resetting UFW rules..."
# sudo ufw reset
