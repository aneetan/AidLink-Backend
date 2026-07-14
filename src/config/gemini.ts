require('dotenv').config();

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_MODEL = 'llama-3.1-8b-instant';
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';

if (!GROQ_API_KEY) {
  throw new Error('Missing GROQ_API_KEY environment variable');
}

const fetchWithRetry = async (url: string, options: RequestInit, retries = 3): Promise<Response> => {
  for (let i = 0; i < retries; i++) {
    const response = await fetch(url, options);

    // 429 = rate limited, wait and retry
    if (response.status === 429) {
      const waitTime = (i + 1) * 5000; // 5s, 10s, 15s
      console.log(`Rate limited, retrying in ${waitTime / 1000}s... (attempt ${i + 1}/${retries})`);
      await new Promise(res => setTimeout(res, waitTime));
      continue;
    }

    return response;
  }
  throw new Error('Request failed after multiple retries');
};

export const getGenerativeModel = () => {
  return {
    generateContent: async (prompt: string) => {
      const response = await fetchWithRetry(GROQ_API_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${GROQ_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: GROQ_MODEL,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 250,
          temperature: 0.6,
          top_p: 0.9
        })
      });

      const responseText = await response.text();

      let data: any;
      try {
        data = JSON.parse(responseText);
      } catch (jsonError: any) {
        throw new Error(`Invalid JSON from Groq: ${jsonError.message}. Body: ${responseText.slice(0, 500)}`);
      }

      if (!response.ok) {
        const message = data?.error?.message || `Groq API error ${response.status}: ${responseText.slice(0, 500)}`;
        throw new Error(message);
      }

      const generatedText = data?.choices?.[0]?.message?.content;
      if (typeof generatedText !== 'string') {
        throw new Error(`Unexpected Groq response format: ${responseText.slice(0, 500)}`);
      }

      return {
        response: { text: () => generatedText }
      };
    }
  };
};