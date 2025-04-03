#!/bin/bash

echo "Updating system and installing UFW..."
sudo apt update && sudo apt install ufw -y  # For Ubuntu/Debian-based systems

# Set default policies: Deny all incoming traffic, allow all outgoing traffic
echo "Setting default policies..."
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow OpenSSH for remote access
echo "Allowing SSH connections (Rate-Limited)..."
sudo ufw allow OpenSSH
sudo ufw limit OpenSSH  # Prevent brute-force attacks

# Allow port for the TTS API (Restrict to trusted IPs if needed)
TTS_PORT=5000
TRUSTED_IP="YOUR_IP_ADDRESS"  # Replace with a real trusted IP if needed

echo "Allowing TTS API traffic on port $TTS_PORT..."
sudo ufw allow $TTS_PORT/tcp  # Open port for API access

# Uncomment the next line to restrict API access to a specific IP
# echo "Restricting API access to trusted IP $TRUSTED_IP..."
# sudo ufw allow from $TRUSTED_IP to any port $TTS_PORT

# Allow HTTPS for secure web access
echo "Allowing HTTPS traffic on port 443..."
sudo ufw allow 443/tcp

# Block unnecessary ports to reduce attack surface
echo "Blocking unused ports..."
sudo ufw deny 80/tcp   # Block HTTP (force HTTPS usage)
sudo ufw deny 21/tcp   # Block FTP
sudo ufw deny 23/tcp   # Block Telnet
sudo ufw deny 3389/tcp # Block RDP

# Rate limit API requests to prevent abuse (adjust as needed)
echo "Setting rate limits for API..."
sudo ufw limit $TTS_PORT/tcp

# Enable UFW logging for security monitoring
echo "Enabling UFW logging..."
sudo ufw logging on

# Enable the firewall
echo "Enabling UFW firewall..."
sudo ufw enable

# Display the current firewall status
echo "Current firewall status:"
sudo ufw status verbose

# Optional: Reset firewall rules if needed (commented out for safety)
# echo "Resetting UFW rules..."
# sudo ufw reset
