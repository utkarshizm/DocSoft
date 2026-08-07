import asyncio
import time
import httpx
from statistics import median

API_URL = "https://docsoft-tueu.onrender.com/ask"
HEADERS = {
    "Authorization": "Bearer sk_docsoft_user_001",
    "Content-Type": "application/json"
}
PAYLOAD = {"question": "What is my name?"}

TOTAL_REQUESTS = 12
CONCURRENCY = 1  # Run sequentially to avoid Google Gemini rate limits

async def send_request(client, request_num):
    start_time = time.perf_counter()
    try:
        response = await client.post(API_URL, headers=HEADERS, json=PAYLOAD, timeout=60.0)
        response.raise_for_status()
        elapsed = time.perf_counter() - start_time
        print(f"Request {request_num+1}: {elapsed:.2f}s (Status: {response.status_code})")
        return elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"Request {request_num+1}: FAILED in {elapsed:.2f}s - {e}")
        return None

async def main():
    print(f"Starting load test: {TOTAL_REQUESTS} requests, {CONCURRENCY} concurrent...")
    latencies = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def limited_request(client, i):
        async with semaphore:
            res = await send_request(client, i)
            await asyncio.sleep(1) # 1 second delay to respect 15 RPM limit
            return res

    async with httpx.AsyncClient() as client:
        tasks = [limited_request(client, i) for i in range(TOTAL_REQUESTS)]
        results = await asyncio.gather(*tasks)
        
    latencies = [r for r in results if r is not None]
    
    if not latencies:
        print("All requests failed!")
        return

    latencies.sort()
    p50 = median(latencies)
    p95_index = int(len(latencies) * 0.95) - 1
    p95 = latencies[p95_index] if p95_index >= 0 else latencies[-1]
    
    print("\n" + "="*40)
    print("LOAD TEST RESULTS")
    print("="*40)
    print(f"Total Successful Requests: {len(latencies)}/{TOTAL_REQUESTS}")
    print(f"p50 latency: {p50:.2f}s")
    print(f"p95 latency: {p95:.2f}s")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
