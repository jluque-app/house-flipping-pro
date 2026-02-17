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
        'filter.city': 'Ciudad',
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
        'common.loading': 'Cargando...',
        'city.barcelona': 'Barcelona',
        'city.valencia': 'Valencia',
        'city.madrid': 'Madrid',
        'city.madrid_soon': 'Madrid (Próximamente)',
        'tutorial.title_1': 'Bienvenido a House Flipping Pro',
        'tutorial.desc_1': 'Tu herramienta profesional para encontrar oportunidades de House Flipping en Barcelona.',
        'tutorial.title_2': 'Analiza el ROI',
        'tutorial.desc_2': 'Cada propiedad incluye un cálculo estimado de rentabilidad basado en costes de reforma y precios de venta de la zona.',
        'tutorial.title_3': 'Entiende el Gap',
        'tutorial.desc_3': "El 'Gap' te indica el margen de negociación. Buscamos propiedades donde el Valor de Inversión (VI) supere al Valor de Oferta (VO).",
        'tutorial.next': 'Siguiente',
        'tutorial.start': 'Comenzar a Invertir',
        'tutorial.skip': 'Saltar Tutorial',
    },
    en: {
        'app.title': 'House Flipping Pro',
        'app.subtitle': 'Smart Real Estate Investment',
        'nav.map': 'Map',
        'nav.ranking': 'Ranking',
        'nav.filters': 'Filters',
        'nav.clear': 'Clear',
        'filter.city': 'City',
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
        'common.loading': 'Loading...',
        'city.barcelona': 'Barcelona',
        'city.valencia': 'Valencia',
        'city.madrid': 'Madrid',
        'city.madrid_soon': 'Madrid (Coming Soon)',
        'tutorial.title_1': 'Welcome to House Flipping Pro',
        'tutorial.desc_1': 'Your professional tool to find House Flipping opportunities in Barcelona.',
        'tutorial.title_2': 'Analyze ROI',
        'tutorial.desc_2': 'Each property includes an estimated profitability calculation based on renovation costs and local market prices.',
        'tutorial.title_3': 'Understand the Gap',
        'tutorial.desc_3': '"The Gap" indicates the negotiation margin. We look for properties where Investment Value (IV) exceeds Offer Value (OV).',
        'tutorial.next': 'Next',
        'tutorial.start': 'Start Investing',
        'tutorial.skip': 'Skip Tutorial',
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
