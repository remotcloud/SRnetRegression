import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

for fig in xrange(1, figure().number): ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()