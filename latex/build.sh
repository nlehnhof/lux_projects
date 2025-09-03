FNAME=main

# OPTION `build.sh genall`: Generate all auxiliary files
if [ ${1:-not} = genall ]
then

    # Remove all auxiliary files
    sh clean.sh

    # Pre-compilation
    pdflatex ${FNAME}.tex

    # Generate index files
    makeindex ${FNAME}.nlo -s nomencl.ist -o ${FNAME}.nls

    # Generate .blg .bbl bibliography files
    bibtex ${FNAME}

fi

# Pre-compile again to add bibliography to TOC
pdflatex ${FNAME}.tex

# Final compilation
pdflatex ${FNAME}.tex

# Remove auxiliary files
sh clean.sh keepaux
