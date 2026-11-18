from flask import Flask, request, redirect, session
import secrets
import urllib.parse
import requests
import os
import json
import csv 

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

CLIENT_KEY = os.getenv("CLIENT_KEY")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = "https://crate-folk-quarterly.ngrok-free.dev/auth/tiktok/callback"

@app.route("/")
def home():
    return '<a href="/login">Connect TikTok</a>'

@app.route("/login")
def login():
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state

    params = {
        "client_key": CLIENT_KEY,
        "scope": "user.info.basic,video.list",
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "state": state
    }

    url = "https://www.tiktok.com/v2/auth/authorize/?" + urllib.parse.urlencode(params)
    return redirect(url)

@app.route("/auth/tiktok/callback")
def callback():
    if request.args.get("state") != session.get("oauth_state"):
        return "State mismatch"

    code = request.args.get("code")

    response = requests.post(
        "https://open.tiktokapis.com/v2/oauth/token/",
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "client_key": CLIENT_KEY,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI
        }
    )

    token_data = response.json()

    # Do NOT display the actual token in the browser
    if "access_token" in token_data:
        session["access_token"] = token_data["access_token"]
        return '<h1>Access token received successfully!</h1><a href="/videos">Get my videos</a>'

    return token_data

@app.route("/videos")
def videos():
    access_token = session.get("access_token")

    all_videos = []
    cursor = None

    while True:
        body = {
            "max_count": 20
        }

        if cursor is not None:
            body["cursor"] = cursor

        response = requests.post(
            "https://open.tiktokapis.com/v2/video/list/",
            params={
                "fields": "id,create_time,title,video_description,duration,share_url,like_count,comment_count,share_count,view_count"
            },
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json=body
        )

        result = response.json()

        if "data" not in result:
            return result

        data = result["data"]

        print("VIDEOS THIS PAGE:", len(data.get("videos", [])))
        print("HAS MORE:", data.get("has_more"), "CURSOR:", data.get("cursor"))

        all_videos.extend(data.get("videos", []))

        if not data.get("has_more", False):
            break

        cursor = data["cursor"]

    #Save the complete video history as JSON
    with open("tiktok_videos.json", "w", encoding="utf-8") as f:
        json.dump(all_videos, f, indent=2, ensure_ascii=False)

    #Save the same data as CSV
    if all_videos:
        with open("tiktok_videos.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_videos[0].keys()
                )
                
            writer.writeheader()
            writer.writerows(all_videos)

    return {
            "message": "Tiktok video data saved successfully",
            "count": len(all_videos)
        }

if __name__ == "__main__":
    app.run(port=5000, debug=True)