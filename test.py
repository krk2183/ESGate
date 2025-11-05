# import requests
# import json

# # =================================================================
# # === 1. PASTE YOUR EXACT MISTRAL API KEY IN THE LINE BELOW ===
# # =================================================================
# MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY_HERE"
# # =================================================================

# # --- Configuration ---
# MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
# HEADERS = {
#     "Authorization": f"Bearer {MISTRAL_API_KEY}",
#     "Content-Type": "application/json"
# }
# payload = {
#     "model": "mistral-small-latest",
#     "messages": [{"role": "user", "content": "Say hello!"}]
# }

# print("--- Attempting to connect to Mistral API... ---")

# if MISTRAL_API_KEY == "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":
#     print("\n!!! ERROR: You have not replaced the placeholder API key. Please edit the script and add your key. !!!")
# else:
#     try:
#         response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload)
        
#         # This function will raise an error if the status code is 4xx or 5xx
#         response.raise_for_status() 
        
#         print("\n--- ✅ SUCCESS! ---")
#         print("API key is valid and the request was successful.")
#         print("API Response:")
#         print(response.json())

#     except requests.exceptions.HTTPError as e:
#         print(f"\n--- ❌ FAILURE ---")
#         print(f"Request failed with status code: {e.response.status_code}")
        
#         if e.response.status_code == 401:
#             print("This is a '401 Unauthorized' error. It confirms the problem is with your API key or account.")
#             print("\nThings to double-check:")
#             print("1. Copy/Paste: Ensure there are NO extra spaces or characters before or after your key.")
#             print("2. Account Status: Log in to your Mistral AI dashboard to confirm your account is active and has credits/a valid payment method.")
#             print("3. Key Permissions: Ensure the key has the necessary permissions to access the chat completions API.")
        
#         print("\nFull server response:")
#         print(e.response.text)

#     except Exception as e:
#         print(f"\n--- An unexpected error occurred ---")
#         print(str(e))

import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATA (Using the final combined risk score) ---
# Assuming the result from the previous normalization is:
current_risk_score = 72.0 

# --- 2. GAUGE PARAMETERS ---
MAX_VALUE = 100
# Define colors for risk zones (normalized to 0-100)
# (Start Angle, End Angle, Color)
risk_zones = [
    (0, 30, 'green'),      # Low Risk
    (30, 60, 'gold'),      # Medium Risk / Caution
    (60, 100, 'darkred')   # High Risk / Alert
]

# --- 3. PLOTTING ---
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

# A. Draw the Gauge Background Arcs (Risk Zones)
for start, end, color in risk_zones:
    # Convert normalized scores to radians (0 to pi)
    start_rad = np.deg2rad(start / MAX_VALUE * 180)
    end_rad = np.deg2rad(end / MAX_VALUE * 180)
    
    # Create a small arc to represent the color zone
    ax.bar(x=start_rad + (end_rad - start_rad) / 2, 
           height=0.5, # Thickness of the gauge band
           width=end_rad - start_rad,
           bottom=0.5, # Position of the gauge band
           color=color,
           edgecolor='white',
           linewidth=2
          )

# B. Plot the Pointer (Needle)
# Map the current score (0-100) to radians (0 to pi)
current_angle = np.deg2rad(current_risk_score / MAX_VALUE * 180)

# The plot function creates a line from the center (r=0) to the edge
ax.plot([0, current_angle], [0, 1], 
        color='black', 
        linewidth=3, 
        marker='^', markersize=10)


# C. Aesthetics
ax.set_theta_zero_location("W") # Start the gauge on the left
ax.set_theta_direction(1) # Go counter-clockwise (0 to 180 degrees)

# Hide radial axis (r-axis) and angular axis (theta-axis)
ax.set_rlabel_position(0)
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.spines['polar'].set_visible(False)

# D. Add Title and Value Text
ax.text(0.5, 0.4, "Combined Risk Score", transform=fig.transFigure, 
        fontsize=14, color='gray', ha='center')
ax.text(0.5, 0.5, f"{current_risk_score:.1f}", transform=fig.transFigure, 
        fontsize=40, fontweight='bold', color='black', ha='center', va='center')

plt.show()