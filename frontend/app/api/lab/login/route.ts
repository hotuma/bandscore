import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export const runtime = 'nodejs';

export async function POST(req: Request) {
    try {
        const body = await req.json();
        const { accessKey } = body;

        const CORRECT_KEY = process.env.LAB_ACCESS_KEY;
        console.log("[Login Debug] LAB_ACCESS_KEY loaded:", CORRECT_KEY ? "YES" : "NO", "(Value length: " + (CORRECT_KEY?.length || 0) + ")");

        // Safety check for server configuration
        if (!CORRECT_KEY) {
            console.error('[Login Debug] LAB_ACCESS_KEY is missing from process.env');
            return NextResponse.json(
                { error: 'System configuration error' },
                { status: 500 }
            );
        }

        if (accessKey === CORRECT_KEY) {
            // Set cookie
            (await cookies()).set({
                name: 'lab_access',
                value: '1',
                httpOnly: true,
                sameSite: 'lax',
                secure: process.env.NODE_ENV === 'production',
                path: '/',
                maxAge: 60 * 60 * 24 * 7, // 1 week
            });

            return NextResponse.json({ ok: true });
        } else {
            return NextResponse.json(
                { error: 'Invalid Access Key' },
                { status: 401 }
            );
        }
    } catch (err) {
        console.error('Login error:', err);
        return NextResponse.json(
            { error: 'Internal Server Error' },
            { status: 500 }
        );
    }
}
