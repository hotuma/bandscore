'use client';

import React, { useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { addToWaitlist } from '../actions/waitlist';

// Option C: Placeholder for Stripe Payment Link
const STRIPE_PAYMENT_LINK_URL = "https://buy.stripe.com/test_xxxxxxxxx"; // Replace with actual live/test link

function WaitlistContent() {
    const searchParams = useSearchParams();
    const rawFrom = searchParams.get('from');
    const from = rawFrom === 'export' ? 'export' : rawFrom === 'limit' ? 'limit' : 'none';

    const [email, setEmail] = useState('');
    const [pain, setPain] = useState('');
    const [songsPerMonth, setSongsPerMonth] = useState('');
    const [isSubmitted, setIsSubmitted] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [errorDetails, setErrorDetails] = useState<string | null>(null);
    const [isAlreadyRegistered, setIsAlreadyRegistered] = useState(false);

    // Logging: WAITLIST_VIEW
    React.useEffect(() => {
        console.log("WAITLIST_VIEW", { from });
    }, [from]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setErrorDetails(null);
        setIsAlreadyRegistered(false);

        // Client-side validation
        if (pain.trim().length < 3) {
            console.log("WAITLIST_SUBMIT_INVALID", { field: "pain" });
            setErrorDetails('「今いちばん困っていること」を1行で入力してください');
            setIsLoading(false);
            return;
        }

        try {
            const result = await addToWaitlist({
                email,
                pain,
                songsPerMonth: songsPerMonth || undefined,
                fromSource: from,
                site: ""
            });

            if (result.success) {
                console.log("WAITLIST_SUBMIT_SUCCESS", { from, songsPerMonth: songsPerMonth || null });
                setIsSubmitted(true);
            } else if (result.alreadyRegistered) {
                console.log("WAITLIST_SUBMIT_SUCCESS", { from, songsPerMonth: songsPerMonth || null, status: "already_registered" });
                setIsAlreadyRegistered(true);
                setIsSubmitted(true);
            } else {
                setErrorDetails(result.error || 'エラーが発生しました');
            }
        } catch (err) {
            setErrorDetails('予期せぬエラーが発生しました');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
            <div className="max-w-md w-full bg-white rounded-2xl shadow-xl overflow-hidden">
                <div className="p-8 space-y-6">
                    <div className="text-center space-y-2">
                        <h1 className="text-2xl font-bold text-gray-900">
                            {isSubmitted
                                ? (isAlreadyRegistered ? 'すでに受付済みです' : '受付しました')
                                : (from === 'export'
                                    ? '作った下書きを、DAW に持っていきませんか？'
                                    : (from === 'limit'
                                        ? 'この作業、まだ続けますよね？'
                                        : '正式版の先行案内を受け取る'))}
                        </h1>
                        <p className="text-gray-600 text-sm leading-relaxed">
                            {isSubmitted
                                ? (isAlreadyRegistered ? 'このメールアドレスは既に登録されています。' : '公開準備が整い次第、ご案内します。')
                                : (from === 'export'
                                    ? 'Early Access の優先案内をお送りします。Export（JSON/TXT）で制作に持ち込めます。'
                                    : (from === 'limit'
                                        ? 'Early Access の優先案内をお送りします。解析回数の制限を解除して、下書きを量産できます。'
                                        : '公開準備が整い次第、メールでお知らせします。'))}
                        </p>
                    </div>

                    {!isSubmitted ? (
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label htmlFor="pain" className="block text-sm font-medium text-gray-700 mb-1">
                                    今いちばん困っていること（1行・必須）
                                </label>
                                <textarea
                                    id="pain"
                                    name="pain"
                                    required
                                    rows={2}
                                    disabled={isLoading}
                                    className="appearance-none rounded-lg relative block w-full px-4 py-3 border border-gray-300 placeholder-gray-400 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:bg-gray-100 disabled:text-gray-500 text-sm"
                                    placeholder="例：耳コピの答え合わせがつらい"
                                    value={pain}
                                    onChange={(e) => setPain(e.target.value)}
                                />
                            </div>

                            <div>
                                <label htmlFor="songsPerMonth" className="block text-sm font-medium text-gray-700 mb-1">
                                    月に何曲くらい耳コピ/採譜しますか？（任意）
                                </label>
                                <select
                                    id="songsPerMonth"
                                    name="songsPerMonth"
                                    disabled={isLoading}
                                    className="appearance-none rounded-lg relative block w-full px-4 py-3 border border-gray-300 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:bg-gray-100 disabled:text-gray-500 text-sm bg-white"
                                    value={songsPerMonth}
                                    onChange={(e) => setSongsPerMonth(e.target.value)}
                                >
                                    <option value="">選択しない</option>
                                    <option value="0-1">0-1曲</option>
                                    <option value="2-5">2-5曲</option>
                                    <option value="6-10">6-10曲</option>
                                    <option value="11+">11曲以上</option>
                                </select>
                            </div>

                            <div>
                                <label htmlFor="email" className="sr-only">Email address</label>
                                <input
                                    id="email"
                                    name="email"
                                    type="email"
                                    required
                                    disabled={isLoading}
                                    className="appearance-none rounded-lg relative block w-full px-4 py-3 border border-gray-300 placeholder-gray-400 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:bg-gray-100 disabled:text-gray-500"
                                    placeholder="メールアドレス"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                />
                            </div>

                            {errorDetails && (
                                <div className="text-red-600 text-sm text-center bg-red-50 p-2 rounded border border-red-100">
                                    {errorDetails}
                                </div>
                            )}

                            <div className="text-xs text-center text-gray-400">
                                しつこい連絡はしません。案内は Early Access の募集時のみです。
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading}
                                className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-bold text-white transition-all ${isLoading
                                    ? 'bg-blue-400 cursor-wait'
                                    : 'bg-blue-600 hover:bg-blue-700 active:scale-95'
                                    }`}
                            >
                                {isLoading ? '送信中...' : '優先案内を受け取る'}
                            </button>
                        </form>
                    ) : (
                        <div className="space-y-4">
                            <div className={`p-4 rounded-lg text-center text-sm font-medium border ${isAlreadyRegistered ? 'bg-yellow-50 text-yellow-700 border-yellow-100' : 'bg-green-50 text-green-700 border-green-100'
                                }`}>
                                {isAlreadyRegistered
                                    ? 'ご登録ありがとうございます。案内をお待ちください。'
                                    : '登録ありがとうございます！\nサービス開始をお待ちください。'}
                            </div>

                            {/* Option C: Purchase CTA */}
                            {(from === 'export' || from === 'limit') && (
                                <div className="space-y-2">
                                    <a
                                        href={STRIPE_PAYMENT_LINK_URL}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="block w-full text-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-colors"
                                        onClick={() => console.log("EA_PURCHASE_CLICK", { from })}
                                    >
                                        Early Access を購入する（¥1,980）
                                    </a>
                                    <div className="text-xs text-center text-gray-500 space-y-1">
                                        <p>※ Early Access は開発中です。解析精度は保証されません。</p>
                                        <p>※ 買い切り・返金可（7日以内）</p>
                                    </div>
                                </div>
                            )}

                            <a
                                href="/early-access"
                                className={`block w-full text-center font-bold py-3 px-4 rounded-lg transition-colors ${(from === 'export' || from === 'limit')
                                    ? 'bg-gray-100 hover:bg-gray-200 text-gray-700' // downgrade visual priority if purchase button exists
                                    : 'bg-blue-600 hover:bg-blue-700 text-white' // keep primary if no purchase button
                                    }`}
                            >
                                Early Access に戻る
                            </a>

                            <a
                                href="/demo"
                                className="block w-full text-center text-gray-400 hover:text-gray-600 text-xs font-medium transition-colors"
                            >
                                デモに戻る
                            </a>
                        </div>
                    )}
                </div>

                {/* Footer decoration */}
                <div className="bg-gray-50 p-4 text-center">
                    <p className="text-xs text-gray-400">BandScore - AI Guitar Tab</p>
                </div>
            </div>
        </main>
    );
}

export default function WaitlistPage() {
    return (
        <Suspense fallback={null}>
            <WaitlistContent />
        </Suspense>
    );
}
