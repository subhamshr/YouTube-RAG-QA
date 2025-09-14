from urllib.parse import urlparse, parse_qs

def get_youtube_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.hostname in ["youtu.be"]:
        return parsed_url.path.lstrip("/")
    else:
        raise ValueError("Invalid YouTube URL")
