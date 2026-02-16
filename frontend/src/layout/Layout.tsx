import { useState, useEffect } from 'react';
import Sidebar from '../components/Sidebar';
import MapPage from '../pages/MapPage';
import RankingPage from '../pages/RankingPage';
import { AuthProvider } from '../contexts/AuthContext';
import { FilterProvider } from '../contexts/FilterContext';
import { LanguageProvider } from '../contexts/LanguageContext';
import OnboardingModal from '../components/OnboardingModal';

function Dashboard() {
    const [view, setView] = useState<'map' | 'ranking'>('map');
    const [isOnboardingOpen, setIsOnboardingOpen] = useState(false);

    useEffect(() => {
        const hasSeen = localStorage.getItem('hasSeenOnboarding');
        if (!hasSeen) setIsOnboardingOpen(true);
    }, []);

    const handleCloseOnboarding = () => {
        setIsOnboardingOpen(false);
        localStorage.setItem('hasSeenOnboarding', 'true');
    };

    return (
        <div className="flex h-screen w-screen overflow-hidden bg-gray-100">
            <Sidebar
                view={view}
                onViewChange={setView}
                onOpenHelp={() => setIsOnboardingOpen(true)}
            />
            <OnboardingModal
                isOpen={isOnboardingOpen}
                onClose={handleCloseOnboarding}
            />

            <main className="flex-1 relative h-full">
                {view === 'map' ? (
                    <MapPage />
                ) : (
                    <RankingPage />
                )}
            </main>
        </div>
    );
}

export default function Layout() {
    return (
        <AuthProvider>
            <LanguageProvider>
                <FilterProvider>
                    <Dashboard />
                </FilterProvider>
            </LanguageProvider>
        </AuthProvider>
    );
}
