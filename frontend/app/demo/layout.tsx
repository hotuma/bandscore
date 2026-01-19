import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: {
        absolute: "ギター曲コード解析デモ｜開発中プレビュー",
    },
    description: "開発中のギター曲コード解析ロジックを確認できるデモです。一曲まるごとのコード進行をプレビューできます。",
};

export default function DemoLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <>{children}</>;
}
