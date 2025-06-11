import { H1, H2, P } from "@/components/ui/typography";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Mail, MapPin, Calendar, GraduationCap } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function AboutPage() {
  const skills = {
    languages: ["Python", "C++", "Java", "JavaScript", "TypeScript"],
    tools: ["TensorFlow", "OpenCV", "AWS", "Docker", "Git"],
    specialties: ["Machine Learning", "Computer Vision", "Deep Learning", "Natural Language Processing"]
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="mb-12">
        <H1 className="text-3xl md:text-4xl font-bold mb-4">About Me</H1>
        <P className="text-lg text-muted-foreground">
          Passionate about leveraging technology to solve complex problems and create meaningful impact.
        </P>
      </div>

      {/* Personal Info Card */}
      <Card className="mb-8">
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1 space-y-4">
              <div className="flex items-center gap-2 text-sm">
                <MapPin className="h-4 w-4 text-muted-foreground" />
                <span>Based in [Your Location]</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Mail className="h-4 w-4 text-muted-foreground" />
                <a href="mailto:contact@saikanam.com" className="hover:text-primary transition-colors">
                  contact@saikanam.com
                </a>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <GraduationCap className="h-4 w-4 text-muted-foreground" />
                <span>Computer Science Graduate</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Calendar className="h-4 w-4 text-muted-foreground" />
                <span>Available for opportunities</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Bio Section */}
      <div className="prose prose-lg dark:prose-invert max-w-none mb-12">
        <P className="leading-relaxed">
          I am Saik, a humble wanderer in these Lands Between. Schooled in the arcane arts
          of Computer Science, I've toiled in realms called Intel, NASA, and Allegro. I wield the tongues of
          Python, C++, and more, crafting modest tools to aid our fractured world. Though my skills are but a
          candle to the darkness that surrounds us, I offer them in service to our cause.
        </P>
        <P className="leading-relaxed">
          Now, I stand at the crossroads of fate, a Tarnished seeking purpose. I offer what meager skills I possess
          in service to our cause. Should you find use for one such as I, I would be honored to walk this perilous
          path alongside you.
        </P>
      </div>

      {/* Skills Section */}
      <div className="space-y-8 mb-12">
        <div>
          <H2 className="text-2xl font-bold mb-4">Technical Skills</H2>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Languages</h3>
              <div className="flex flex-wrap gap-2">
                {skills.languages.map((skill) => (
                  <Badge key={skill} variant="secondary" className="text-sm">
                    {skill}
                  </Badge>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Tools & Frameworks</h3>
              <div className="flex flex-wrap gap-2">
                {skills.tools.map((tool) => (
                  <Badge key={tool} variant="outline" className="text-sm">
                    {tool}
                  </Badge>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Specialties</h3>
              <div className="flex flex-wrap gap-2">
                {skills.specialties.map((specialty) => (
                  <Badge key={specialty} className="text-sm">
                    {specialty}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <Card className="bg-muted/50">
        <CardContent className="pt-6">
          <div className="text-center space-y-4">
            <H2 className="text-2xl font-bold">Let's Connect</H2>
            <P className="text-muted-foreground">
              I'm always interested in new opportunities and collaborations.
            </P>
            <div className="flex flex-wrap gap-4 justify-center">
              <Button asChild>
                <a href="mailto:contact@saikanam.com">Get in Touch</a>
              </Button>
              <Button variant="outline" asChild>
                <a href="/resume.pdf" target="_blank">Download Resume</a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 