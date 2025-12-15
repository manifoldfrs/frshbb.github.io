import Layout from '@/components/Layout';
import Image from 'next/image';

export default function About() {
  return (
    <Layout title="About" url="/about">
      <div className="max-w-3xl mx-auto">
        <div className="space-y-8">
          {/* Header */}
          <section className="text-center">
            <h1 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              Velvet Steel
            </h1>
          </section>
          {/* Content */}
          <section className="prose prose-nord max-w-none">
            <div className="space-y-6 text-text-primary leading-relaxed">
              <p>
                You&apos;ll find reflections on everything from programming to
                health and fitness. From religious contemplation, to practical
                philoosophy, and poetry. The most interesting insights often
                emerge from the cross-pollination of ideas.
              </p>

              <h2>Values</h2>
              <ul>
                <li>Curiosity</li>
                <li>
                  Respect
                </li>
                <li>
                  Progress
                </li>
                <li>Agency</li>
                <li>Conviction</li>
              </ul>
            </div>
          </section>

          {/* Currently Listening */}
          <section className="bg-background-darker rounded-lg p-6">
            <div className="w-full">
              <iframe
                style={{ borderRadius: '12px' }}
                src="https://open.spotify.com/embed/track/7exOrewyAiFxxHkGMXIvUn?si=2ce2892e10ee4d18"
                width="100%"
                height="352"
                allowFullScreen
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"
              />
            </div>
          </section>

          {/* Contact */}
          <section className="bg-background-darker rounded-lg p-6">
            <h2 className="text-xl font-semibold text-text-primary mb-4">
              Get in Touch
            </h2>
            <div className="space-y-2 text-text-secondary">
              <p>
                <strong>Email:</strong>{' '}
                <a
                  href="mailto:faris@duck.com"
                  className="text-links hover:text-primary transition-colors"
                >
                  faris@duck.com
                </a>
              </p>
              <p>
                <strong>Location:</strong> San Francisco, CA
              </p>
            </div>
          </section>
        </div>
      </div>
    </Layout>
  );
}
