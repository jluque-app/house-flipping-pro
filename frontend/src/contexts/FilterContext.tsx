import React, { createContext, useContext, useState } from 'react';

export interface Filters {
    city?: string;
    bbox?: string;
    neighborhood?: string;
    price_min?: number;
    price_max?: number;
    roi_min?: number;
    amenities?: string[];
}

interface FilterContextType {
    filters: Filters;
    setFilters: React.Dispatch<React.SetStateAction<Filters>>;
    updateFilter: (key: keyof Filters, value: any) => void;
    resetFilters: () => void;
}

const FilterContext = createContext<FilterContextType | undefined>(undefined);

export function FilterProvider({ children }: { children: React.ReactNode }) {
    const [filters, setFilters] = useState<Filters>({});

    const updateFilter = (key: keyof Filters, value: any) => {
        setFilters(prev => ({ ...prev, [key]: value }));
    };

    const resetFilters = () => {
        setFilters({});
    };

    return (
        <FilterContext.Provider value={{ filters, setFilters, updateFilter, resetFilters }}>
            {children}
        </FilterContext.Provider>
    );
}

export function useFilters() {
    const context = useContext(FilterContext);
    if (context === undefined) {
        throw new Error('useFilters must be used within a FilterProvider');
    }
    return context;
}
