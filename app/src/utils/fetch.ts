/**
 * fetch with exponential backoff — retries on 5xx or network errors.
 */
export async function fetchWithRetry(
  url: string,
  maxRetries = 5,
  baseDelayMs = 1000
): Promise<Response> {
  let delay = baseDelayMs;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const r = await fetch(url);
      if (r.ok || r.status < 500) return r;
      if (attempt < maxRetries - 1) await new Promise(res => setTimeout(res, delay));
      delay = Math.min(delay * 2, 8000);
    } catch {
      if (attempt < maxRetries - 1) await new Promise(res => setTimeout(res, delay));
      delay = Math.min(delay * 2, 8000);
    }
  }
  return fetch(url);
}
