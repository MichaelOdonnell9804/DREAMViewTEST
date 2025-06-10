import ROOT
import os
import sys
# from CMSPLOTS import CMS_lumi
# from CMSPLOTS import tdrstyle
import CMS_lumi
import tdrstyle


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[1;31m'
    UNDERLINE = '\033[4m'


def DumpHist(hist):
    # dump the number of events in histogram
    print("histogram %s with %d bins" % (hist.GetName, hist.GetNbinsX()))
    for ibin in range(0, hist.GetNbinsX()+2):
        print(" %d bin with %.2f events" % (ibin, hist.GetBinContent(ibin)))


# many helper functions; trimmed for brevity

def DrawHistos(myhistos, mylabels, xmin, xmax, xlabel, ymin, ymax, ylabel, outputname, dology=True, showratio=False, dologx=False, lheader=None, donormalize=False, binomialratio=False, yrmax=2.0, yrmin=0.0, yrlabel=None, MCOnly=False, leftlegend=False, mycolors=None, legendPos=None, legendNCols=1, linestyles=None, markerstyles=None, showpull=False, doNewman=False, doPearson=False, ignoreHistError=False, ypullmin=-3.99, ypullmax=3.99, drawashist=False, padsize=(2, 0.9, 1.1), setGridx=False, setGridy=False, drawoptions=None, legendoptions=None, ratiooptions=None, dologz=False, doth2=False, ratiobase=0, redrawihist=-1, extraText=None, noCMS=False, noLumi=False, nMaxDigits=None, addOverflow=False, addUnderflow=False, plotdiff=False, hratiopanel=None, doratios=None, hpulls=None, W_ref=600, is5TeV=False, outdir="plots", savepdf=True, zmin=0, zmax=2, extralabels=None, extralheader=None, extraToDraw=None, exlegoffset=0.08):
    """Draw histograms with the CMS tdr style."""
    if mycolors is None:
        mycolors = []
    if legendPos is None:
        legendPos = []
    if linestyles is None:
        linestyles = []
    if markerstyles is None:
        markerstyles = []
    if drawoptions is None:
        drawoptions = []
    if legendoptions is None:
        legendoptions = []
    if ratiooptions is None:
        ratiooptions = []

    tdrstyle.setTDRStyle()
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetPaintTextFormat(".3f")

    CMS_lumi.lumi_sqrtS = ""
    CMS_lumi.relPosX = 0.25
    CMS_lumi.extraText = ""
    if MCOnly:
        CMS_lumi.extraText = "Simulation"
    if isinstance(extraText, str):
        CMS_lumi.extraText = extraText

    if ymax is None:
        ymax = max([h.GetMaximum() for h in myhistos]) * 1.25
    if ymin is None:
        ymin = min([h.GetMinimum() for h in myhistos]) * 0.75

    canvas = ROOT.TCanvas("c2"+outputname, "c2", 50, 50, W_ref, 600)
    if dology:
        canvas.SetLogy()
    if dologx:
        canvas.SetLogx()
    if dologz:
        canvas.SetLogz()
    canvas.SetLeftMargin(0.15)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.06)

    h1 = ROOT.TH2F("h2" + outputname, "h2", 80, xmin, xmax, 80, ymin, ymax)
    h1.GetZaxis().SetLabelSize(0.03)
    if zmin is not None and zmax is not None:
        h1.GetZaxis().SetRangeUser(zmin, zmax)
    h1.GetYaxis().SetTitle(ylabel)
    h1.GetXaxis().SetTitle(xlabel)
    h1.Draw()

    legend = ROOT.TLegend(0.7, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetFillColor(0)
    legend.SetNColumns(legendNCols)
    if lheader:
        legend.SetHeader(lheader)

    for idx, ihisto in enumerate(myhistos):
        ihcolone = ihisto.Clone(f"{ihisto.GetName()}_Clone")
        if idx < len(mycolors):
            ihcolone.SetLineColor(mycolors[idx])
            ihcolone.SetMarkerColor(mycolors[idx])
        ihcolone.SetLineWidth(3)
        ihcolone.Draw(drawoptions[idx] + " same" if idx < len(drawoptions) else "same")
        if idx < len(mylabels):
            legend.AddEntry(ihcolone, str(mylabels[idx]), "LE")

    legend.Draw()
    if not noLumi:
        CMS_lumi.CMS_lumi(canvas, 0, 0, plotCMS=not noCMS)
    canvas.Update()

    dirpath = os.path.join(outdir)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if savepdf:
        canvas.Print(f"{outdir}/{outputname}.pdf")

