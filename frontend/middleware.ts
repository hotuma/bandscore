import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
    const { pathname } = request.nextUrl;

    // Only protect /lab routes
    if (pathname.startsWith('/lab')) {
        // Exception for login page and its children (e.g. actions)
        if (pathname.startsWith('/lab/login')) {
            return NextResponse.next();
        }

        // Check for access cookie
        // Note: In middleware, we use request.cookies
        const hasAccess = request.cookies.has('lab_access');

        if (!hasAccess) {
            // Redirect to login page
            const loginUrl = new URL('/lab/login', request.url);
            return NextResponse.redirect(loginUrl);
        }
    }

    return NextResponse.next();
}

export const config = {
    matcher: [
        /*
         * Match all request paths starting with /lab
         */
        '/lab/:path*',
    ],
};
