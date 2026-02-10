#!/usr/bin/env bash
# =============================================================================
# RNSR Windows Setup Script
#
# Run this in Git Bash (bundled with Git for Windows) to set up and launch
# the RNSR demo on a Windows machine.
#
# Usage:
#   bash setup_windows.sh            # Full setup + launch
#   bash setup_windows.sh --no-run   # Setup only, don't launch the demo
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours (Git Bash supports ANSI)
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
NO_RUN=false
for arg in "$@"; do
    case "$arg" in
        --no-run) NO_RUN=true ;;
        --help|-h)
            echo "Usage: bash setup_windows.sh [--no-run] [--help]"
            echo "  --no-run   Set up the environment but don't launch the demo"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  RNSR - Windows Setup                       ${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""

# ---------------------------------------------------------------------------
# 1. Locate Python >= 3.10
# ---------------------------------------------------------------------------
info "Checking for Python >= 3.10 ..."

PYTHON_CMD=""

# Try common commands: py (Windows launcher), python, python3
for cmd in py python python3; do
    if command -v "$cmd" &>/dev/null; then
        # For the 'py' launcher, request 3.10+
        if [ "$cmd" = "py" ]; then
            PY_VER=$("$cmd" -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        else
            PY_VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        fi

        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)

        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ] 2>/dev/null; then
            if [ "$cmd" = "py" ]; then
                PYTHON_CMD="py -3"
            else
                PYTHON_CMD="$cmd"
            fi
            ok "Found $cmd -> Python $PY_VER"
            break
        else
            warn "$cmd is Python $PY_VER (need >= 3.10)"
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    err "Python >= 3.10 is required but was not found on PATH."
    echo ""
    echo "  Install Python from https://www.python.org/downloads/"
    echo "  Make sure to check 'Add Python to PATH' during installation."
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create / activate virtual environment
# ---------------------------------------------------------------------------
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in $VENV_DIR ..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    ok "Virtual environment created."
else
    ok "Virtual environment already exists at $VENV_DIR."
fi

# Activate (Git Bash uses Scripts/ not bin/)
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    err "Could not find activation script in $VENV_DIR."
    exit 1
fi
ok "Virtual environment activated."

# ---------------------------------------------------------------------------
# 3. Upgrade pip & install Python dependencies
# ---------------------------------------------------------------------------
info "Upgrading pip ..."
python -m pip install --upgrade pip --quiet

info "Installing RNSR dependencies from requirements.txt ..."
python -m pip install -r requirements.txt --quiet
ok "Python dependencies installed."

# ---------------------------------------------------------------------------
# 4. Check / install system dependencies (optional)
# ---------------------------------------------------------------------------
echo ""
info "Checking optional system dependencies ..."

# --- Poppler (needed by pdf2image for OCR fallback) ---
POPPLER_OK=false
if command -v pdftoppm &>/dev/null; then
    ok "Poppler (pdftoppm) found on PATH."
    POPPLER_OK=true
else
    warn "Poppler not found. pdf2image OCR fallback will not work without it."
    echo ""
    echo "  Poppler converts PDF pages to images for OCR."
    echo "  It is only needed if your PDFs require OCR (scanned pages)."
    echo ""

    if command -v winget &>/dev/null; then
        read -rp "  Install Poppler via winget now? [y/N] " yn
        case "$yn" in
            [Yy]*)
                info "Installing Poppler via winget ..."
                winget install oschwartz10612.Poppler --accept-source-agreements --accept-package-agreements 2>/dev/null && POPPLER_OK=true
                if $POPPLER_OK; then
                    ok "Poppler installed. You may need to restart your terminal for PATH changes."
                else
                    warn "Poppler install failed. You can install manually later:"
                    echo "    winget install oschwartz10612.Poppler"
                fi
                ;;
            *)
                info "Skipping Poppler. Install later if needed:"
                echo "    winget install oschwartz10612.Poppler"
                ;;
        esac
    else
        info "winget not available. To install Poppler manually:"
        echo "    1. Download from https://github.com/oschwartz10612/poppler-windows/releases"
        echo "    2. Extract and add the bin/ folder to your PATH"
    fi
fi

# --- Tesseract (needed by pytesseract for OCR) ---
TESSERACT_OK=false
if command -v tesseract &>/dev/null; then
    ok "Tesseract found on PATH."
    TESSERACT_OK=true
else
    warn "Tesseract not found. pytesseract OCR will not work without it."
    echo ""
    echo "  Tesseract is an OCR engine used as a fallback for scanned PDFs."
    echo ""

    if command -v winget &>/dev/null; then
        read -rp "  Install Tesseract via winget now? [y/N] " yn
        case "$yn" in
            [Yy]*)
                info "Installing Tesseract via winget ..."
                winget install UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements 2>/dev/null && TESSERACT_OK=true
                if $TESSERACT_OK; then
                    ok "Tesseract installed. You may need to restart your terminal for PATH changes."
                else
                    warn "Tesseract install failed. You can install manually later:"
                    echo "    winget install UB-Mannheim.TesseractOCR"
                fi
                ;;
            *)
                info "Skipping Tesseract. Install later if needed:"
                echo "    winget install UB-Mannheim.TesseractOCR"
                ;;
        esac
    else
        info "winget not available. To install Tesseract manually:"
        echo "    1. Download from https://github.com/UB-Mannheim/tesseract/wiki"
        echo "    2. Run the installer and add it to your PATH"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Set up .env file
# ---------------------------------------------------------------------------
echo ""
if [ -f ".env" ]; then
    ok ".env file already exists."
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        ok "Created .env from .env.example."
    else
        warn ".env.example not found. Creating a minimal .env file ..."
        cat > .env <<'ENVEOF'
# RNSR Environment Variables
# Set at least one LLM provider API key below.

GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

LLM_PROVIDER=auto
LOG_LEVEL=INFO
ENVEOF
        ok "Created minimal .env file."
    fi

    echo ""
    echo -e "${YELLOW}  ╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}  ║  IMPORTANT: Edit .env and add at least one API key      ║${NC}"
    echo -e "${YELLOW}  ║                                                          ║${NC}"
    echo -e "${YELLOW}  ║    GOOGLE_API_KEY=your-key     (recommended)             ║${NC}"
    echo -e "${YELLOW}  ║    OPENAI_API_KEY=your-key                               ║${NC}"
    echo -e "${YELLOW}  ║    ANTHROPIC_API_KEY=your-key                            ║${NC}"
    echo -e "${YELLOW}  ╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
fi

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  Setup Complete                              ${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""
echo "  Python:     $(python --version 2>&1)"
echo "  Venv:       $VENV_DIR (activated)"
echo "  Poppler:    $(if $POPPLER_OK; then echo 'installed'; else echo 'NOT installed (OCR fallback disabled)'; fi)"
echo "  Tesseract:  $(if $TESSERACT_OK; then echo 'installed'; else echo 'NOT installed (OCR fallback disabled)'; fi)"
echo ""

# Check if an API key is set
if grep -qE '^(GOOGLE_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)=.+' .env 2>/dev/null; then
    # At least one key has a non-empty value (but it might be the placeholder)
    HAS_KEY=true
    # Exclude placeholder values
    if grep -qE '=your_.*_key_here' .env 2>/dev/null && ! grep -qE '^(GOOGLE_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)=[^y]' .env 2>/dev/null; then
        HAS_KEY=false
    fi
else
    HAS_KEY=false
fi

if ! $HAS_KEY; then
    warn "No API key detected in .env -- the demo will warn on startup."
    echo "  Edit .env and set at least one of: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
    echo ""
fi

# ---------------------------------------------------------------------------
# 7. Launch (unless --no-run)
# ---------------------------------------------------------------------------
if $NO_RUN; then
    info "Skipping launch (--no-run). To start the demo later, run:"
    echo ""
    echo "    source $VENV_DIR/Scripts/activate"
    echo "    python demo.py"
    echo ""
else
    info "Launching RNSR demo ..."
    echo "  Open http://localhost:7860 in your browser."
    echo ""
    python demo.py
fi
