import requests

response = requests.post("https://journal-server-vqkc.onrender.com/journal", json={
    "date": "2026-03-22",
    "entries": [
        {"time": "09:14", "transcript": "so i woke up late today and missed my morning run"},
        {"time": "12:30", "transcript": "had lunch with priya we talked about the trip to goa"},
        {"time": "21:00", "transcript": "feeling pretty good today overall got a lot of work done"}
    ]
})

print("Status:", response.status_code)
print("Response:", response.text)