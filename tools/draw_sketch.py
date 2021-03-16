import os
import ndjson
import cv2 as cv
import numpy as np
from math import log
from operator import truediv
from scipy.special import comb
import xml.etree.ElementTree as xml


white = (256,256,256)
gray = (100,100,100)
colors = [
    (255,0,0),
    (255,255,0),
    (255,0,255),
    (0,255,255),
    (0,255,0),
    (0,0,255),
    (150,150,100) ,
    (100,48,186) ,
    (255,100,  0) ,
    (50,175,0) ,
    (78, 130, 225) ,
    (128, 223, 81) ,
    (234, 53, 144) ,
    (128, 76, 226) ,
    (46, 146, 237) ,
    (174, 155, 99) ,
    (255, 29, 143) ,
    (215, 148, 63) ,
    (11, 161, 253) ,
]

color_list = np.array(colors)[...,::-1]

def drawColor(sketch, imgSizes, zoomTimes):
    penColor = white
    canvas = np.zeros((imgSizes[0],imgSizes[1],3),dtype='uint8')
    for stroke in sketch:
        penColor = colors[int(stroke[2][0])]
        cv.circle(canvas, 
                  (int(stroke[0][0]*zoomTimes), int(stroke[1][0]*zoomTimes)), 
                  4, penColor)
        for i in range(1, len(stroke[0])):
            penColor = colors[int(stroke[2][i])]
            cv.line(canvas, 
                    (int(stroke[0][i-1]*zoomTimes), int(stroke[1][i-1]*zoomTimes)), 
                    (int(stroke[0][i]*zoomTimes), int(stroke[1][i]*zoomTimes)), 
                    penColor)
            cv.circle(canvas, 
                      (int(stroke[0][i]*zoomTimes), int(stroke[1][i]*zoomTimes)), 
                      2, gray)
    return canvas


def writesvg(strokelist, labelnamelist, path, colorlist, ifendpoint=False):
    stroke_list = []
    label_list = [[] for i in range(len(labelnamelist))]
    order = 0
    for stroke in strokelist:
        new_stroke = [[],[]]
        last_label = stroke[2][0]
        for pi in range(len(stroke[0])):
            if stroke[2][pi] == last_label:
                new_stroke[0].append(stroke[0][pi])
                new_stroke[1].append(stroke[1][pi])
            else:
                stroke_list.append(new_stroke)
                label_list[last_label].append(order)
                order += 1
                last_label = stroke[2][pi]
                new_stroke = [[stroke[0][pi-1], stroke[0][pi]],[stroke[1][pi-1], stroke[1][pi]]]
        stroke_list.append(new_stroke)
        label_list[last_label].append(order)
        order += 1
    sketch2svg.convert(stroke_list, label_list, labelnamelist, path, colorlist, ifendpoint=ifendpoint)



class sketch2svg(object):
    @staticmethod
    def convert(stroke_list, label_list, label_name_list, out_file_path_str, label_colors=None, ifendpoint=False, ifPoints=False):
        """
        To inkscape svg file
        :param stroke_list: a list of strokes
        :param label_list: a list of label groupings
        :param label_name_list: label name corresponding to each index in label_list
        :param out_file_path_str:
        :param label_colors:
        :return:
        """
        image_width = 256
        image_height = 256

        sketch_name = out_file_path_str.split('\\')[-1].split('.')[0]

        node_svg = sketch2svg._get_node_svg(sketch_name, image_width, image_height)

        # Meta info
        node_svg.append(sketch2svg._get_node_metadata())
        node_svg.append(sketch2svg._get_node_namedview())
        node_svg.append(sketch2svg._get_node_defs())

        # Each stroke grouping
        node_g_root = sketch2svg._get_node_g_root()

        g_idx_offset = 200
        g_path_idx_offset = 200
        for lidx, stroke_indices in enumerate(label_list):
            label_strokes = [stroke_list[sidx] for sidx in stroke_indices]
            if len(label_strokes):
                label_color = label_colors[lidx] if label_colors is not None else None

                node_g_root.append(sketch2svg._get_node_g_label(g_idx_offset,
                                                                          label_name_list[lidx],
                                                                          g_path_idx_offset,
                                                                          label_strokes,
                                                                          color=label_color,
                                                                          ifendpoint=ifendpoint, 
                                                                          ifPoints=ifPoints))
                g_idx_offset += 1
                g_path_idx_offset += len(label_strokes)

        node_svg.append(node_g_root)

        tree = xml.ElementTree(node_svg)
        tree.write(out_file_path_str, encoding='utf-8', xml_declaration=True)

    @staticmethod
    def _get_node_svg(docname, view_width, view_height):
        node_svg = xml.Element('svg')
        node_svg.attrib['xmlns:dc'] = 'http://purl.org/dc/elements/1.1/'
        node_svg.attrib['xmlns:cc'] = 'http://creativecommons.org/ns#'
        node_svg.attrib['xmlns:rdf'] = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        node_svg.attrib['xmlns:svg'] = 'http://www.w3.org/2000/svg'
        node_svg.attrib['xmlns'] = 'http://www.w3.org/2000/svg'
        node_svg.attrib['xmlns:sodipodi'] = 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
        node_svg.attrib['xmlns:inkscape'] = 'http://www.inkscape.org/namespaces/inkscape'
        node_svg.attrib['height'] = '100%'
        node_svg.attrib['version'] = '1.1'
        node_svg.attrib['viewBox'] = '0 0 {} {}'.format(view_width, view_height)
        node_svg.attrib['width'] = '100%'
        node_svg.attrib['id'] = 'svg100'
        node_svg.attrib['sodipodi:docname'] = docname
        node_svg.attrib['inkscape:version'] = '0.92.2 (5c3e80d, 2017-08-06)'
        return node_svg

    @staticmethod
    def _get_node_metadata():
        metadata = xml.Element('metadata')
        metadata.attrib['id'] = 'metadata100'
        rdfRDF = xml.Element('rdf:RDF')

        ccWork = xml.Element('cc:Work')
        ccWork.attrib['rdf:about'] = ''

        dcformat = xml.SubElement(ccWork, 'dc:format')
        dcformat.text = 'image/svg+xml'
        dctype = xml.SubElement(ccWork, 'dc:type',
                                attrib={'rdf:resource': 'http://purl.org/dc/dcmitype/StillImage'})
        dctitle = xml.SubElement(ccWork, 'dc:title')

        rdfRDF.append(ccWork)
        metadata.append(rdfRDF)

        return metadata

    @staticmethod
    def _get_node_namedview():
        namedview = xml.Element('sodipodi:namedview')
        namedview.attrib['pagecolor'] = '#ffffff'
        namedview.attrib['bordercolor'] = '#666666'
        namedview.attrib['borderopacity'] = '1'
        namedview.attrib['objecttolerance'] = '10'
        namedview.attrib['gridtolerance'] = '10'
        namedview.attrib['guidetolerance'] = '10'
        namedview.attrib['inkscape:pageopacity'] = '0'
        namedview.attrib['inkscape:pageshadow'] = '2'
        namedview.attrib['inkscape:window-width'] = '640'
        namedview.attrib['inkscape:window-height'] = '480'
        namedview.attrib['id'] = 'namedview100'
        namedview.attrib['showgrid'] = 'false'
        namedview.attrib['inkscape:zoom'] = '3.3164062'
        namedview.attrib['inkscape:cx'] = '100'
        namedview.attrib['inkscape:cy'] = '100'
        namedview.attrib['inkscape:current-layer'] = 'g100'
        return namedview

    @staticmethod
    def _get_node_defs():
        node_defs = xml.Element('defs')
        node_defs.attrib['id'] = 'defs100'
        return node_defs

    @staticmethod
    def _get_node_g_root():
        g_root = xml.Element('g')
        g_root.attrib['id'] = 'g100'
        g_root.attrib['style'] = 'fill:none;stroke:#000000;stroke-linecap:round;stroke-linejoin:round'
        return g_root

    @staticmethod
    def _get_node_g_label(g_id_offset, label_name, path_id_offset, stroke_list, color=None, ifendpoint=False, ifPoints=False):
        stroke_width = 3
        point_r = 3
        g_label = xml.Element('g')
        g_label.attrib['id'] = 'g' + str(g_id_offset)
        g_label.attrib['inkscape:label'] = label_name
        g_label.attrib['sodipodi:insensitive'] = 'true'
        if color is None:
            g_label.attrib['style'] = 'display:inline;stroke:#000000;stroke-opacity:1;stroke-width:' + str(stroke_width)
        else:
            color_hex = '#%02x%02x%02x' % (color[0], color[1], color[2])
            g_label.attrib['style'] = 'display:inline;stroke:{};stroke-opacity:1;stroke-width:{}'.format(color_hex, stroke_width)

        # Append each stroke to this label group
        for idx, stroke in enumerate(stroke_list):
            d_str = 'M {} {}'.format(stroke[0][0], stroke[1][0])
            for pidx in range(1, len(stroke[0])):
                d_str += ' L {},{}'.format(stroke[0][pidx], stroke[1][pidx])

            g_path = xml.Element('path')
            g_path.attrib['inkscape:connector-curvature'] = '0'
            g_path.attrib['id'] = 'path' + str(path_id_offset + idx)
            g_path.attrib['d'] = d_str
            g_label.append(g_path)
            
            if ifendpoint:
                g_start = xml.Element('circle')
                g_start.attrib['cx'] = str(stroke[0][0])
                g_start.attrib['cy'] = str(stroke[1][0])
                g_start.attrib['r'] = str(point_r)
                g_start.attrib['id'] = 'startpoint' + str(path_id_offset + idx)
                g_label.append(g_start)

                # g_end = xml.Element('circle')
                # g_end.attrib['cx'] = str(stroke[0][-1])
                # g_end.attrib['cy'] = str(stroke[1][-1])
                # g_end.attrib['r'] = str(point_r)
                # g_end.attrib['id'] = 'endpoint' + str(path_id_offset + idx)
                # g_label.append(g_end)

            if ifPoints:
                for pidx in range(0, len(stroke[0])):
                    g_p = xml.Element('circle')
                    g_p.attrib['cx'] = str(stroke[0][pidx])
                    g_p.attrib['cy'] = str(stroke[1][pidx])
                    g_p.attrib['r'] = str(point_r)
                    g_p.attrib['id'] = 'p' + str(idx)
                    g_label.append(g_p)
        return g_label