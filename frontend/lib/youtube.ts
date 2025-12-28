export function extractVideoId(url: string): string | null {
    try {
        const parsed = new URL(url);
        const hostname = parsed.hostname;

        // Handle youtube.com/watch?v=ID
        if (hostname === 'www.youtube.com' || hostname === 'youtube.com') {
            if (parsed.pathname === '/watch') {
                return parsed.searchParams.get('v');
            }
            // Handle youtube.com/shorts/ID
            if (parsed.pathname.startsWith('/shorts/')) {
                return parsed.pathname.split('/shorts/')[1];
            }
            // Handle youtube.com/embed/ID (just in case)
            if (parsed.pathname.startsWith('/embed/')) {
                return parsed.pathname.split('/embed/')[1];
            }
        }

        // Handle youtu.be/ID
        if (hostname === 'youtu.be') {
            return parsed.pathname.slice(1);
        }

        return null;
    } catch (e) {
        // If URL parsing fails, try regex or just return null
        return null;
    }
}
