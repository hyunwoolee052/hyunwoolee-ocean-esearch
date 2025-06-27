import urllib.parse
import urllib.request

data = {}
data["id"] = "sooList"
data["key"] = "qPwOeIrU-2506-KUVBOC-1211"  # expired 2026-06-25
data["sdate"] = "19620101"  # YYYYMMDD
data["edate"] = "20241231"  # YYYYMMDD
url_values = urllib.parse.urlencode(data)
url = f"https://www.nifs.go.kr/bweb/OpenAPI_json"
full_url = f"{url}?{url_values}"