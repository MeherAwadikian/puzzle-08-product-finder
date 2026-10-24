export interface Env {
  ASSETS: Fetcher;
  LENS_UPLOADS: R2Bucket;
  R2_PUBLIC_URL: string;
  SERPAPI_KEY: string;
  OPENROUTER_API_KEY: string;
}

interface VisualMatch {
  position?: number;
  title?: string;
  link?: string;
  source?: string;
  thumbnail?: string;
  price?: { value?: string; extracted_value?: number; currency?: string };
}

interface CleanedItem {
  name: string;
  price?: string;
  source?: string;
  link?: string;
  thumbnail?: string;
}

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/api/search" && request.method === "POST") {
      return handleSearch(request, env, ctx);
    }

    return env.ASSETS.fetch(request);
  },
};

async function handleSearch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  try {
    const formData = await request.formData();
    const file = formData.get("image");
    if (!file || !(file instanceof File)) {
      return new Response(JSON.stringify({ error: "No image uploaded" }), { status: 400, headers: JSON_HEADERS });
    }
    if (file.size > 8 * 1024 * 1024) {
      return new Response(JSON.stringify({ error: "Image too large (max 8MB)" }), { status: 400, headers: JSON_HEADERS });
    }

    const ext = (file.type.split("/")[1] || "jpg").replace("jpeg", "jpg");
    const key = `uploads/${crypto.randomUUID()}.${ext}`;

    await env.LENS_UPLOADS.put(key, await file.arrayBuffer(), {
      httpMetadata: { contentType: file.type || "image/jpeg" },
    });

    const publicImageUrl = `${env.R2_PUBLIC_URL}/${key}`;

    // Step 1: real Google Lens reverse-image product search via SerpApi
    const lensUrl = `https://serpapi.com/search.json?engine=google_lens&url=${encodeURIComponent(publicImageUrl)}&api_key=${env.SERPAPI_KEY}`;
    const lensRes = await fetch(lensUrl);
    if (!lensRes.ok) {
      const t = await lensRes.text();
      return new Response(JSON.stringify({ error: "Google Lens lookup failed", detail: t }), { status: 502, headers: JSON_HEADERS });
    }
    const lensData: any = await lensRes.json();

    // Clean up the temp upload now that Lens has fetched it
    ctx.waitUntil(env.LENS_UPLOADS.delete(key).catch(() => {}));

    const visualMatches: VisualMatch[] = lensData.visual_matches || [];
    const rawMatchCount = visualMatches.length;

    if (rawMatchCount === 0) {
      return new Response(
        JSON.stringify({ rawMatchCount: 0, cleaned: [], cleanupModel: "none", note: "Google Lens returned no visual matches for this image." }),
        { headers: JSON_HEADERS }
      );
    }

    // Keep only candidates that actually look like shoppable products (have a price or a shopping-like source)
    const candidates = visualMatches.slice(0, 30).map((m) => ({
      title: m.title || "",
      source: m.source || "",
      link: m.link || "",
      thumbnail: m.thumbnail || "",
      price: m.price?.value || (m.price?.extracted_value ? `${m.price.extracted_value}` : ""),
    }));

    // Step 2: AI cleanup - normalize/dedupe/rank raw Lens matches into a presentable shortlist
    const { cleaned, modelUsed } = await cleanupWithAI(candidates, env);

    return new Response(
      JSON.stringify({ rawMatchCount, cleaned, cleanupModel: modelUsed }),
      { headers: JSON_HEADERS }
    );
  } catch (err: any) {
    return new Response(JSON.stringify({ error: "Unexpected error", detail: String(err?.message || err) }), {
      status: 500,
      headers: JSON_HEADERS,
    });
  }
}

async function cleanupWithAI(
  candidates: { title: string; source: string; link: string; thumbnail: string; price: string }[],
  env: Env
): Promise<{ cleaned: CleanedItem[]; modelUsed: string }> {
  const prompt = `You are cleaning up raw Google Lens visual-match results for a "product finder" app.
Below is a JSON array of raw candidate matches (title, source, link, thumbnail, price - price may be empty).

Task:
1. Discard matches that are clearly not purchasable products (e.g. Reddit threads, Wikipedia, news articles, forums, generic informational pages).
2. Merge near-duplicate entries (same product from the same or different retailer) into a single cleaned entry, normalizing the product title into a short, human-readable name (strip SEO junk, site names, and excess punctuation).
3. Prefer entries that have a price and a real shopping/retailer source.
4. Rank the output with the most likely exact match first.
5. Return AT MOST 8 results.

Return ONLY a raw JSON array (no markdown, no prose) of objects shaped exactly like:
[{"name": "...", "price": "..." or null, "source": "...", "link": "...", "thumbnail": "..."}]

Raw candidates:
${JSON.stringify(candidates)}`;

  const models = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
  ];

  for (const model of models) {
    try {
      const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${env.OPENROUTER_API_KEY}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.2,
        }),
      });
      if (!res.ok) continue;
      const data: any = await res.json();
      const text: string = data.choices?.[0]?.message?.content || "";
      const jsonMatch = text.match(/\[[\s\S]*\]/);
      if (!jsonMatch) continue;
      const parsed = JSON.parse(jsonMatch[0]);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return { cleaned: parsed.slice(0, 8), modelUsed: model };
      }
    } catch {
      // try next model
    }
  }

  // Fallback: no LLM available, do a basic local cleanup instead of failing the whole request
  const seen = new Set<string>();
  const fallback: CleanedItem[] = [];
  for (const c of candidates) {
    const norm = c.title.toLowerCase().slice(0, 40);
    if (!c.price || seen.has(norm)) continue;
    seen.add(norm);
    fallback.push({ name: c.title, price: c.price, source: c.source, link: c.link, thumbnail: c.thumbnail });
    if (fallback.length >= 8) break;
  }
  return { cleaned: fallback, modelUsed: "fallback-local (no LLM reachable)" };
}
