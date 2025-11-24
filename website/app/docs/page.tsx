import React from 'react';
import { BookOpen, Code, Download, FileText, Video, Zap } from 'lucide-react';
import Link from 'next/link';

export default function DocsPage() {
  const resources = [
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: "Getting Started",
      description: "Learn how to set up and use OncoVision AI in your workflow.",
      link: "/docs/getting-started"
    },
    {
      icon: <Code className="w-6 h-6" />,
      title: "API Reference",
      description: "Complete API documentation for integrating with our platform.",
      link: "/docs/api"
    },
    {
      icon: <Download className="w-6 h-6" />,
      title: "Download Center",
      description: "Get the latest versions of our software and tools.",
      link: "/docs/downloads"
    },
    {
      icon: <FileText className="w-6 h-6" />,
      title: "White Papers",
      description: "In-depth technical documentation and research papers.",
      link: "/docs/whitepapers"
    },
    {
      icon: <Video className="w-6 h-6" />,
      title: "Video Tutorials",
      description: "Step-by-step video guides for all features.",
      link: "/docs/tutorials"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white py-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Documentation
            </span>
          </h1>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto">
            Everything you need to get started with OncoVision AI and make the most of our platform.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {resources.map((resource, index) => (
            <Link 
              key={index}
              href={resource.link}
              className="group bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6 hover:bg-slate-800/70 transition-all duration-300 hover:border-blue-500/30 flex flex-col"
            >
              <div className="w-12 h-12 flex items-center justify-center rounded-xl bg-blue-500/10 text-blue-400 mb-4 group-hover:bg-blue-500/20 transition-colors">
                {resource.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2 group-hover:text-blue-400 transition-colors">
                {resource.title}
              </h3>
              <p className="text-slate-400 mb-4 flex-grow">
                {resource.description}
              </p>
              <div className="text-blue-400 group-hover:text-blue-300 transition-colors flex items-center">
                Learn more
                <svg className="w-4 h-4 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-16 bg-slate-800/50 border border-slate-700/50 rounded-2xl p-8">
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-500/10 text-blue-400 mb-6 mx-auto">
              <Zap className="w-8 h-8" />
            </div>
            <h3 className="text-2xl font-semibold mb-4">Need help getting started?</h3>
            <p className="text-slate-300 mb-6">
              Our support team is here to help you with any questions or issues you might have.
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <Link
                href="/contact"
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 rounded-xl font-medium transition-all duration-300 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/50 flex items-center justify-center"
              >
                Contact Support
              </Link>
              <Link
                href="/"
                className="px-6 py-3 bg-slate-700/50 hover:bg-slate-700/70 border border-slate-600/50 rounded-xl font-medium transition-all duration-300 flex items-center justify-center"
              >
                Back to Home
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
