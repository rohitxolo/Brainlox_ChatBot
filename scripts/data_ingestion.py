import requests
import json
from bs4 import BeautifulSoup

# Define the URL
url = "https://brainlox.com/courses/category/technical"

# Set a User-Agent header to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

# Make the request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    courses = soup.find_all("div", class_="single-courses-box")
    
    extracted_data = []
    for course in courses:
        title_tag = course.find("h3").find("a")
        title = title_tag.text.strip() if title_tag else "No Title"
        desc_tag = course.find("p")
        description = desc_tag.text.strip() if desc_tag else "No Description"
        course_link = "https://brainlox.com" + title_tag["href"] if title_tag else "No Link"
        
        extracted_data.append({
            "title": title,
            "description": description,
            "link": course_link
        })
    
    # Save extracted data to JSON
    with open("data/extracted_data.json", "w", encoding="utf-8") as file:
        json.dump(extracted_data, file, indent=4, ensure_ascii=False)
    
    print("Data extracted and saved successfully.")
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")
