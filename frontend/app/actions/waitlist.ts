'use server';

import { createClient } from '@supabase/supabase-js';

// Environment variables must be set on the server
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

type WaitlistResult = {
    success: boolean;
    alreadyRegistered?: boolean;
    error?: string;
};

export type AddToWaitlistInput = {
    email: string;
    pain: string;
    songsPerMonth?: string;
    fromSource?: string; // "export" | "limit" | "none"
    site?: string; // honeypot
};

export async function addToWaitlist(input: AddToWaitlistInput): Promise<WaitlistResult> {
    const { email, pain, songsPerMonth, fromSource, site } = input;

    // Honeypot check (Spam protection)
    if (site) {
        console.log('Spam honeypot triggered:', email);
        // Return fake success to confuse bots
        return { success: true };
    }

    if (!email || !email.includes('@')) {
        return { success: false, error: '有効なメールアドレスを入力してください' };
    }

    if (!pain || pain.trim().length < 3) {
        return { success: false, error: '「今いちばん困っていること」を1行で入力してください' };
    }

    if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
        console.error('Supabase environment variables missing');
        return { success: false, error: 'システム設定エラーが発生しました' };
    }

    try {
        const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
            auth: {
                persistSession: false,
                autoRefreshToken: false,
                detectSessionInUrl: false,
            },
        });

        // Check availability first or rely on unique constraint
        // User requested "Already registered" message, so we should check error code 23505 (unique_violation)

        const { error } = await supabase
            .from('waitlist')
            .insert({
                email,
                pain: pain.trim(),
                songs_per_month: songsPerMonth ?? null,
                from_source: fromSource ?? null,
            });

        if (error) {
            // Postgres unique violation code: 23505
            if (error.code === '23505') {
                return { success: false, alreadyRegistered: true };
            }
            console.error('Supabase Insert Error:', error);
            throw error;
        }

        return { success: true };

    } catch (err) {
        console.error('Waitlist Action Error:', err);
        return { success: false, error: '登録処理に失敗しました。時間をおいて再度お試しください。' };
    }
}
