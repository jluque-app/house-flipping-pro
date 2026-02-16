import { ArrowRight, TrendingUp, Map, BarChart3, ShieldCheck, Globe } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';

export default function LandingPage() {
    const { t, language, setLanguage } = useLanguage();

    return (
        <div className="bg-white min-h-screen font-sans text-gray-900">
            {/* Navbar */}
            <nav className="fixed w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center space-x-2">
                            <div className="bg-blue-600 p-1.5 rounded-lg">
                                <TrendingUp className="text-white w-5 h-5" />
                            </div>
                            <span className="text-xl font-bold tracking-tight text-gray-900">{t('app.title')}</span>
                        </div>
                        <div className="hidden md:flex space-x-8 items-center">
                            <a href="#features" className="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors">Features</a>
                            <button
                                onClick={() => setLanguage(language === 'es' ? 'en' : 'es')}
                                className="flex items-center space-x-1 text-sm text-gray-600 hover:text-blue-600 transition-colors"
                            >
                                <Globe size={16} />
                                <span className="uppercase">{language}</span>
                            </button>
                        </div>
                        <div>
                            <Link
                                to="/app"
                                className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-all shadow-sm hover:shadow-md"
                            >
                                {t('landing.cta')}
                            </Link>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative pt-32 pb-20 lg:pt-40 lg:pb-28 overflow-hidden">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
                    <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900 mb-6 max-w-4xl mx-auto">
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-indigo-600">
                            {t('landing.hero_title')}
                        </span>
                    </h1>
                    <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-500 mb-10">
                        {t('landing.hero_subtitle')}
                    </p>
                    <div className="flex justify-center gap-4">
                        <Link
                            to="/app"
                            className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-full text-white bg-blue-600 hover:bg-blue-700 md:text-lg md:px-10 transition-all shadow-lg hover:shadow-xl hover:-translate-y-1"
                        >
                            Ver Mapa de Oportunidades
                            <ArrowRight className="ml-2 w-5 h-5" />
                        </Link>
                        <a
                            href="#features"
                            className="inline-flex items-center px-8 py-3 border border-gray-300 text-base font-medium rounded-full text-gray-700 bg-white hover:bg-gray-50 md:text-lg md:px-10 transition-all"
                        >
                            Saber Más
                        </a>
                    </div>
                </div>

                {/* Abstract Background Elements */}
                <div className="absolute top-0 left-1/2 w-full -translate-x-1/2 h-full z-0 pointer-events-none opacity-30 overflow-hidden">
                    <div className="absolute top-[20%] left-[20%] w-72 h-72 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl animate-blob"></div>
                    <div className="absolute top-[20%] right-[20%] w-72 h-72 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000"></div>
                    <div className="absolute bottom-[20%] left-[40%] w-72 h-72 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000"></div>
                </div>
            </section>

            {/* Features Grid */}
            <section id="features" className="py-20 bg-gray-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-base font-semibold text-blue-600 tracking-wide uppercase">Características</h2>
                        <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
                            Todo lo que necesitas para invertir con éxito
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
                        <FeatureCard
                            icon={<Map className="w-8 h-8 text-blue-500" />}
                            title="Mapa Interactivo"
                            description="Visualiza miles de propiedades en Barcelona. Filtra por barrio, precio y rentabilidad estimada."
                        />
                        <FeatureCard
                            icon={<BarChart3 className="w-8 h-8 text-purple-500" />}
                            title="Análisis de ROI"
                            description="Algoritmos avanzados calculan el retorno de inversión y el potencial de revalorización automáticamente."
                        />
                        <FeatureCard
                            icon={<ShieldCheck className="w-8 h-8 text-green-500" />}
                            title="Gap & Liquidez"
                            description="Identifica el 'Gap' entre valor de compra y venta, y evalúa la liquidez del mercado en cada zona."
                        />
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="bg-white border-t border-gray-200 py-12">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center">
                    <div className="flex items-center space-x-2 mb-4 md:mb-0">
                        <div className="bg-blue-600 p-1 rounded-md">
                            <TrendingUp className="text-white w-4 h-4" />
                        </div>
                        <span className="text-lg font-bold text-gray-900">{t('app.title')}</span>
                    </div>
                    <p className="text-gray-500 text-sm">
                        &copy; 2026 {t('app.title')}. Todos los derechos reservados.
                    </p>
                </div>
            </footer>
        </div>
    );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
    return (
        <div className="bg-white rounded-2xl p-8 shadow-sm hover:shadow-md transition-shadow border border-gray-100">
            <div className="mb-5 bg-gray-50 w-16 h-16 rounded-xl flex items-center justify-center">
                {icon}
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-3">{title}</h3>
            <p className="text-gray-500 leading-relaxed">
                {description}
            </p>
        </div>
    );
}
