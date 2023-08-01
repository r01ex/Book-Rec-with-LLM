import threading
from .trinitygenerateFORTEST import (
    generate_recommendation_book,
    generate_recommendation_start,
)

threads = list()
texts = ["추리소설 추천해줘", "일본어 공부할 책 추천해줘", "유럽 요리의 역사와 관련된 책 추천해줘"]
for i, text in enumerate(texts):
    device_id = f"cuda:{i}"
    thread = threading.Thread(
        target=generate_recommendation_start, args=(text, device_id)
    )
    threads.append(thread)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
