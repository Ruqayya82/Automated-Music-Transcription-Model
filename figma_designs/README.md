# TranscribeAI Figma Wireframes

This directory contains the UI/UX wireframe designs for the TranscribeAI web application.

## Design Overview

The TranscribeAI interface is designed to be clean, modern, and user-friendly, focusing on simplicity and ease of use for musicians and audio professionals.

## Logo Design

### Logo File

The TranscribeAI logo is located at: `../logo/TransribeAI logo.png`

**Logo Integration:**
- The logo has been integrated into the web application
- Located in: `src/web_app/static/logo.png`
- Displayed in the header of the web interface
- Responsive sizing: max-width 300px

**Usage in Wireframes:**
- Reference the actual logo file when creating Figma designs
- Maintain consistent branding across all mockups
- Use the logo as the primary brand identifier

## Wireframe Pages

### 1. Landing Page / Home
- **Hero Section**: Logo, headline, CTA button
- **Upload Section**: Drag-and-drop file upload area
- **Features Section**: Highlight key capabilities
- **How It Works**: Step-by-step process
- **Footer**: Links and credits

### 2. Upload Interface
- **Header**: Logo and navigation
- **Upload Area**: 
  - Drag-and-drop zone
  - File type indicators
  - Progress feedback
- **Options Panel**: 
  - Model selection (ML vs Traditional)
  - Audio settings
- **Action Buttons**: Transcribe, Cancel

### 3. Processing View
- **Header**: Logo
- **Progress Indicator**: 
  - Animated progress bar
  - Status messages
  - Current step indicator
- **Audio Visualization**: Waveform display (optional)

### 4. Results Page
- **Header**: Logo and navigation
- **Statistics Card**:
  - Number of notes detected
  - Transcription method used
  - Processing time
- **Download Section**:
  - MIDI file download button
  - MusicXML file download button
  - Preview options
- **Actions**: 
  - New transcription button
  - Share button (future feature)

### 5. Error State
- **Header**: Logo
- **Error Message**: Clear, friendly error explanation
- **Suggestions**: What to try next
- **Action Buttons**: Try again, Upload new file

## Color Palette

```
Primary Colors:
- Indigo:     #6366f1 (Primary actions, links)
- Purple:     #8b5cf6 (Accents, secondary)
- Dark Blue:  #4f46e5 (Hover states)

Semantic Colors:
- Success:    #10b981 (Completed states)
- Error:      #ef4444 (Error messages)
- Warning:    #f59e0b (Warnings)
- Info:       #3b82f6 (Info messages)

Neutral Colors:
- Background: #f9fafb (Page background)
- Card:       #ffffff (Card backgrounds)
- Text:       #111827 (Primary text)
- Secondary:  #6b7280 (Secondary text)
- Border:     #e5e7eb (Borders, dividers)
```

## Typography

**Font Family**: Inter, -apple-system, BlinkMacSystemFont, Segoe UI

**Font Sizes:**
- Heading 1: 2.5rem (40px) - Bold
- Heading 2: 1.5rem (24px) - Semibold
- Heading 3: 1.125rem (18px) - Semibold
- Body: 1rem (16px) - Regular
- Small: 0.875rem (14px) - Regular

## Spacing System

Based on 8px grid:
- xs: 4px
- sm: 8px
- md: 16px
- lg: 24px
- xl: 32px
- 2xl: 48px

## Components

### Buttons
- **Primary**: Indigo background, white text, rounded corners
- **Secondary**: White background, gray border, gray text
- **Download**: Green background, white text

### Cards
- White background
- Subtle shadow
- Rounded corners (12px)
- Padding: 32px

### Input Fields
- Border: 2px solid gray
- Focus: Border changes to indigo
- Rounded corners (8px)

## Responsive Breakpoints

- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

## Accessibility

- Minimum contrast ratio: 4.5:1 for text
- Focus indicators on all interactive elements
- Alt text for all images
- Keyboard navigation support
- Screen reader friendly labels

## Design Tools

**Recommended Tools:**
- Figma (Primary design tool)
- Adobe XD (Alternative)
- Sketch (Alternative)

## Implementation Notes

The actual web application (separate from Figma wireframes) is built with:
- HTML5
- CSS3 (Custom styling)
- JavaScript (Vanilla JS)
- Flask (Backend)

## Files in This Directory

- `logo-concepts.md`: Detailed logo design concepts
- `wireframes-desktop.md`: Desktop wireframe specifications
- `wireframes-mobile.md`: Mobile wireframe specifications
- `design-system.md`: Complete design system documentation
- `assets/`: Directory for exported design assets

## Next Steps

1. Create high-fidelity mockups in Figma
2. Export design assets (logos, icons)
3. Create interactive prototypes
4. User testing and feedback
5. Refinement and iteration

## Notes

- The Figma designs are wireframes/mockups separate from the actual running application
- The web app has its own implementation in `src/web_app/`
- Designers can reference these specifications when creating Figma files
