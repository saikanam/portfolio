import { QuartzComponentConstructor, QuartzComponentProps } from "./types"

interface Options {
  // Add any options you want to be configurable
}

export default ((opts?: Partial<Options>) => {
  function NavBar({ displayClass }: QuartzComponentProps) {
    return (
      <nav className={`navbar ${displayClass ?? ""}`}>
        <div className="nav-buttons">
          <a href="https://saikanam.github.io/portfolio" className="nav-button active">
            <span>Home</span>
          </a>
          <a href="https://saikanam.github.io/portfolio/Projects/" className="nav-button">
            <span>Projects</span>
          </a>
          <a href="#quests" className="nav-button">
            <span>Quests</span>
          </a>
          <a href="#tomes" className="nav-button">
            <span>Tomes</span>
          </a>
          <a href="#alliance" className="nav-button">
            <span>Alliance</span>
          </a>
        </div>
      </nav>
    )
  }

  NavBar.css = `
    .navbar {
      background-color: rgba(22, 22, 24, 0.8);
      backdrop-filter: blur(10px);
      padding: 1rem 0;
      position: sticky;
      top: 0;
      z-index: 1000;
      box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    .nav-buttons {
      display: flex;
      justify-content: center;
      gap: 1.5rem;
    }
    
    .nav-button {
      position: relative;
      background-color: rgba(75, 75, 75, 0.2);
      color: #d4af37;
      border: 2px solid #d4af37;
      padding: 0.7rem 1.5rem;
      text-decoration: none;
      font-family: 'Cinzel', serif;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: 1px;
      transition: all 0.3s ease;
      border-radius: 3px;
      overflow: hidden;
      box-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
    }
    
    .nav-button span {
      position: relative;
      z-index: 1;
    }
    
    .nav-button::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.2), transparent);
      transition: left 0.5s ease;
    }
    
    .nav-button:hover::before {
      left: 100%;
    }
    
    .nav-button::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20"><path fill="%23d4af37" d="M10 1L12.39 6.36L18.06 6.85L13.69 10.84L15.05 16.43L10 13.47L4.95 16.43L6.31 10.84L1.94 6.85L7.61 6.36L10 1Z"/></svg>');
      background-size: 20px 20px;
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    
    .nav-button:hover::after,
    .nav-button.active::after {
      opacity: 0.1;
    }
    
    .nav-button:hover, .nav-button.active {
      background-color: rgba(212, 175, 55, 0.1);
      color: #ffffff;
      transform: translateY(-2px);
      box-shadow: 0 4px 20px rgba(212, 175, 55, 0.5);
    }
    
    @media (max-width: 768px) {
      .nav-buttons {
        flex-wrap: wrap;
      }
      
      .nav-button {
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
      }
    }
  `

  NavBar.afterDOMLoaded = `
    document.addEventListener('nav', () => {
      const navButtons = document.querySelectorAll('.nav-button');
      
      navButtons.forEach(button => {
        button.addEventListener('click', function(e) {
          e.preventDefault();
          navButtons.forEach(btn => btn.classList.remove('active'));
          this.classList.add('active');
          
          console.log(\`Navigating to \${this.getAttribute('href')}\`);
        });
      });
    })
  `

  return NavBar
}) satisfies QuartzComponentConstructor