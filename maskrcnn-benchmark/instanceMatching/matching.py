import os
import sys
from collections import defaultdict
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
import torch
import networkx as nx
from networkx.algorithms import matching
from pycocotools.cocoeval import maskUtils
from utils_davis.evaluation import calc_iou_individual


NODE_NAME = {
    '1':    'a',
    '2':    'b',
    '3':    'c',
    '4':    'd',
    '5':    'e',
    '6':    'f',
    '7':    'g',
    '8':    'h',
    '9':    'i',
    '10':    'j',

}

OBJECT_NAME = {
    'a':    '1',
    'b':    '2',
    'c':    '3',
    'd':    '4',
    'e':    '5',
    'f':    '6',
    'g':    '7',
    'h':    '8',
    'i':    '9',
    'j':    '10',

}


class InstanceMatcher(object):
    def __init__(self, iou_type='bbox'):
        self.iou_type = iou_type

    def __call__(self, prediction, template):

        self.check_inputs(prediction, template)
        self.prepare(prediction, template)
        result = self.matching()
        # matched_prediction = self.select_matched_prediction(prediction, result)

        matched_prediction = self.assign_instance_id(prediction, template, result)

        if len(matched_prediction.bbox) != len(template.bbox):
            a = 1

        # assert len(matched_prediction.bbox) == len(template.bbox), \
        #     "the number of prediction after matched != the number of template"
        return matched_prediction

    @staticmethod
    def check_inputs(predictions, templates):
        """
            The number of instances in predictions must be more than that in templates

        """
        if predictions is None:
            raise Exception('Lack of prediction for matching')
        elif templates is None:
            raise Exception('Lack of template for matching')

        # assert len(predictions) >= len(templates), \
        #     "The number of instances in predictions is less than that in templates"

    def prepare(self, prediction, template):
        if self.iou_type == 'bbox':
            self._prepare_bbox(prediction, template)
        elif self.iou_type == 'segms':
            self._prepare_segms(prediction, template)
        else:
            raise Exception('Lack of iou type for instance matching')

    def _prepare_bbox(self, prediction, template):
        prediction = prediction.convert("xyxy")
        self.predict_objects = prediction.bbox.tolist()
        self.template_objects = template.bbox.tolist()

    def matching(self):
        self._generate_graph()
        node_list, edge_list = self.named_node_list, self.named_edge_list
        matching_result = self._bipartite_maximum_weight(node_list, edge_list)
        result = self._accumulate_results(matching_result)
        return result

    def _generate_graph(self):
        node_list = self._generate_node()
        edge_list = self._generate_edge(node_list)
        self.named_node_list, self.named_edge_list = \
            self._prepare_name_for_matching(node_list, edge_list)
        assert len(self.named_node_list[0]) * len(self.named_node_list[1]) == len(self.named_edge_list), \
            'the number of edges does not match with their corresponding nodes'

    def _generate_node(self):
        predict_objects_list = list(range(len(self.predict_objects)+1))
        # Remove 0. the first object starts from 1
        predict_objects_list.pop(0)
        template_objects_list = list(range(len(self.template_objects)+1))
        template_objects_list.pop(0)
        return predict_objects_list, template_objects_list

    def _generate_edge(self, node_list):
        if self.iou_type == 'bbox':
            return self._generate_edge_bbox(node_list)
        elif self.iou_type == 'segms':
            return self._generate_edge_segms(node_list)
        else:
            raise Exception('Lack of iou type for instance matching')

    def _generate_edge_bbox(self, node_list):
        predict_objects_list, template_objects_list = node_list

        edge_list = []
        for i in predict_objects_list:
            for j in template_objects_list:
                iou = self.__compute_iou(
                    self.predict_objects[i - 1], self.template_objects[j - 1])
                if iou == 0:
                    iou = 0.0000001
                edge_list.append((i, j, {'weight': iou}))
        return edge_list

    def _accumulate_results(self, matching_result):
        """
            Argument:
                matching_result (list(tuple)):
                E.g. [(1,b), (a, 2)]

            Return:
                matched_results (dict)
                ["prediction"]: the index of prediction after matching
                ["template"]:  the index of template after matching

        """
        matched_objects_list = defaultdict(list)
        for (object1, object2) in matching_result:
            if isinstance(object2, str):
                # the index of nodes for matching starts from 1,
                # but the index of prediction starts from 0
                matched_objects_list['predict'].append(object1 - 1)
                object2 = int(OBJECT_NAME[object2])
                matched_objects_list['template'].append(object2 - 1)
                matched_objects_list['weight'].append(self.extract_iou_from_edge(object1, object2))
            else:
                object1 = int(OBJECT_NAME[object1])
                matched_objects_list['template'].append(object1 - 1)
                matched_objects_list['predict'].append(object2 - 1)
                matched_objects_list['weight'].append(self.extract_iou_from_edge(object2, object1))
        return matched_objects_list

    @staticmethod
    def assign_instance_id(prediction, template, matched_objects_list):
        # for id_predict, id_template in \
        #         zip(matched_objects_list['predict'], matched_objects_list['template']):
        #     instance_id = template.get_field("instance_id")[id_template]
        #     prediction[id_predict].add_field("instance_id", instance_id)
        instance_id = template.get_field("instance_id")[matched_objects_list['template']]
        selected_prediction = prediction[matched_objects_list['predict']]
        selected_prediction.add_field("instance_id", instance_id)
        return selected_prediction

    @staticmethod
    def select_matched_prediction(prediction, matched_objects_list):
        return prediction[matched_objects_list['predict']]

    @staticmethod
    def _bipartite_maximum_weight(node_list, edge_list):
        left_node_list = node_list[0]
        right_node_list = node_list[1]
        # Generate Graph
        G = nx.Graph()
        G.add_nodes_from(left_node_list, bipartite=0)
        G.add_nodes_from(right_node_list, bipartite=1)
        G.add_edges_from(edge_list)
        # Check graph
        nodes_1 = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
        nodes_2 = set(G) - nodes_1
        assert set(left_node_list) == nodes_1 and set(right_node_list) == nodes_2, \
            'Fail to generate the Bipartite Graph'
        # Matching
        matching_result = nx.matching.max_weight_matching(G)
        return matching_result

    @staticmethod
    def _prepare_name_for_matching(node_list, edge_list):
        # E.g. node_2: '1' --> 'a'
        node_1 = node_list[0]
        node_2 = [NODE_NAME[str(node)] for node in node_list[1]]
        node_list_named = (node_1, node_2)

        edge_list_named = []
        for edge in edge_list:
            edge_list_named.append((edge[0], NODE_NAME[str(edge[1])], edge[2]))
        return node_list_named, edge_list_named

    @staticmethod
    def __compute_iou(a, b):
        # return maskUtils.iou(d,g,iscrowd)
        return calc_iou_individual(a, b)

    def extract_iou_from_edge(self, object1, object2):
        object2_name = NODE_NAME[str(object2)]
        for edge in self.named_edge_list:
            if edge[0:2] == (object1, object2_name):
                iou = edge[2]['weight']
                return iou
