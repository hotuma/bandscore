"""Test CORS preflight response using raw HTTP"""
import http.client

conn = http.client.HTTPConnection("127.0.0.1", 8000)

# Test 1: OPTIONS preflight
print("=== Test 1: OPTIONS preflight from localhost:3002 ===")
conn.request("OPTIONS", "/analyze/url", headers={
    "Origin": "http://localhost:3002",
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type",
})
resp = conn.getresponse()
print(f"Status: {resp.status} {resp.reason}")
print("Headers:")
for k, v in resp.getheaders():
    print(f"  {k}: {v}")
body = resp.read()
print(f"Body: {body!r}")

# Test 2: OPTIONS preflight from localhost:3000 (should work)
print("\n=== Test 2: OPTIONS preflight from localhost:3000 ===")
conn.request("OPTIONS", "/analyze/url", headers={
    "Origin": "http://localhost:3000",
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type",
})
resp2 = conn.getresponse()
print(f"Status: {resp2.status} {resp2.reason}")
print("Headers:")
for k, v in resp2.getheaders():
    print(f"  {k}: {v}")
body2 = resp2.read()
print(f"Body: {body2!r}")

# Test 3: Simple GET to /test 
print("\n=== Test 3: GET /test with Origin localhost:3002 ===")
conn.request("GET", "/test", headers={
    "Origin": "http://localhost:3002",
})
resp3 = conn.getresponse()
print(f"Status: {resp3.status} {resp3.reason}")
print("Headers:")
for k, v in resp3.getheaders():
    print(f"  {k}: {v}")
body3 = resp3.read()
print(f"Body: {body3!r}")

conn.close()
