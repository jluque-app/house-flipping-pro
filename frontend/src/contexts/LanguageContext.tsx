import { createContext, useContext, useState, type ReactNode } from 'react';

type Language = 'es' | 'en';

interface LanguageContextType {
    language: Language;
    setLanguage: (lang: Language) => void;
    t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

const translations: Record<Language, Record<string, string>> = {
    es: {
        'app.title': 'House Flipping Pro',
        'app.subtitle': 'Inversión Inmobiliaria Inteligente',
        'nav.map': 'Mapa',
        'nav.ranking': 'Ranking',
        'nav.filters': 'Filtros',
        'nav.clear': 'Limpiar',
        'filter.neighborhood': 'Barrio',
        'filter.all_neighborhoods': 'Todos los barrios',
        'filter.price': 'Precio (€)',
        'filter.roi_min': 'ROI Mínimo (%)',
        'nav.help': 'Ayuda / Tour',
        'nav.logout': 'Cerrar Sesión',
        'landing.cta': 'Lanzar App',
        'landing.hero_title': 'Domina el Mercado Inmobiliario e identifica las mejores oportunidades de inversión',
        'landing.hero_subtitle': 'Plataforma líder en inteligencia de datos inmobiliarios para inversores exigentes.',
        'col.property': 'Propiedad',
        'col.price': 'Precio',
        'col.roi': 'ROI',
        'col.gap': 'Gap (Inv/Prop)',
        'col.liquidity': 'Liquidez (Inv)',
        'drawer.maximize': 'Maximizar',
        'drawer.price': 'Precio',
        'drawer.roi': 'ROI Estimado',
        'drawer.gap': 'Gap',
        'drawer.liquidity': 'Liquidez',
    },
    en: {
        'app.title': 'House Flipping Pro',
        'app.subtitle': 'Smart Real Estate Investment',
        'nav.map': 'Map',
        'nav.ranking': 'Ranking',
        'nav.filters': 'Filters',
        'nav.clear': 'Clear',
        'filter.neighborhood': 'Neighborhood',
        'filter.all_neighborhoods': 'All neighborhoods',
        'filter.price': 'Price (€)',
        'filter.roi_min': 'Min ROI (%)',
        'nav.help': 'Help / Tour',
        'nav.logout': 'Logout',
        'landing.cta': 'Launch App',
        'landing.hero_title': 'Master the Real Estate Market and Identify the Best Investment Opportunities',
        'landing.hero_subtitle': 'Leading real estate data intelligence platform for demanding investors.',
        'col.property': 'Property',
        'col.price': 'Price',
        'col.roi': 'ROI',
        'col.gap': 'Gap (Inv/Own)',
        'col.liquidity': 'Liquidity (Inv)',
        'drawer.maximize': 'Maximize',
        'drawer.price': 'Price',
        'drawer.roi': 'Est. ROI',
        'drawer.gap': 'Gap',
        'drawer.liquidity': 'Liquidity',
    }
};

export function LanguageProvider({ children }: { children: ReactNode }) {
    const [language, setLanguage] = useState<Language>('es');

    const t = (key: string) => {
        return translations[language][key] || key;
    };

    return (
        <LanguageContext.Provider value={{ language, setLanguage, t }}>
            {children}
        </LanguageContext.Provider>
    );
}

export function useLanguage() {
    const context = useContext(LanguageContext);
    if (!context) {
        throw new Error('useLanguage must be used within a LanguageProvider');
    }
    return context;
}
