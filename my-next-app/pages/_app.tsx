// Import necessary modules and components
import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import React from 'react';

const MyApp: React.FC<AppProps> = ({ Component, pageProps }: AppProps) => {
  return <Component {...pageProps} />;
};

export default MyApp;
