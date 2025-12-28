import { Metadata } from 'next';

export const metadata: Metadata = {
    robots: {
        index: false,
        follow: false,
    },
};

export default function LabLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <>{children}</>;
}
