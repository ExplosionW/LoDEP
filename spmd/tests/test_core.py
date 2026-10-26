"""CPU checks for structural grouping, fixed teacher memory and prefix equivalence."""
import unittest
import tempfile
import numpy as np
import torch
from torch import nn
from transformers import EsmConfig,EsmModel
from spmd.memory import StructuralMemory
from spmd.models import Student
from spmd.prepare import structural_groups
from spmd.settings import CONFIG,configure_arm,seed

class CoreTests(unittest.TestCase):
    def test_new_student_uses_pretrained_backbone_and_new_identical_heads(self):
        config=EsmConfig(vocab_size=33,hidden_size=320,intermediate_size=640,num_hidden_layers=1,
            num_attention_heads=4,pad_token_id=1,mask_token_id=32,position_embedding_type='rotary')
        original=EsmModel(config)
        with torch.no_grad():original.embeddings.word_embeddings.weight.fill_(.123)
        with tempfile.TemporaryDirectory() as folder:
            original.save_pretrained(folder)
            seed();kd=Student(student_base=folder)
            seed();spmd=Student(student_base=folder)
            torch.testing.assert_close(kd.student.embeddings.word_embeddings.weight,original.embeddings.word_embeddings.weight)
            for key,value in kd.state_dict().items():torch.testing.assert_close(value,spmd.state_dict()[key])
            self.assertGreater(float(kd.mapping.weight.detach().abs().sum()),0.)
            self.assertGreater(float(kd.classifier.weight.detach().abs().sum()),0.)

    def test_structure_clusters_and_teacher_are_fixed(self):
        states=np.asarray([[0]*6,[1]*6,[2]*6,[3]*6])
        groups=structural_groups(states,np.ones_like(states,dtype=float),np.full(4,6),np.asarray([0,0,1,1]))
        self.assertEqual(len(groups['counts']),4)
        torch.manual_seed(42)
        teacher=torch.randn(4,8,requires_grad=True);initial=torch.randn(4,8)
        memory=StructuralMemory(teacher,initial,torch.as_tensor(groups['assignments']),torch.as_tensor(groups['quality']))
        frozen=memory.teacher_geometry.clone();query=torch.randn(4,8,requires_grad=True)
        q,g,updated=memory(query,torch.arange(4));(q+g).backward()
        self.assertTrue(torch.isfinite(query.grad).all());self.assertGreater(float(query.grad.abs().sum()),0.)
        self.assertIsNone(teacher.grad)
        memory.commit(updated);self.assertTrue(torch.equal(memory.teacher_geometry,frozen))
        self.assertFalse(memory.student_bank.requires_grad)

    def test_prefix_equals_full_sequence_and_padding_is_excluded(self):
        torch.manual_seed(42)
        config=EsmConfig(vocab_size=33,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
            pad_token_id=1,mask_token_id=32,position_embedding_type='rotary',hidden_dropout_prob=0.,attention_probs_dropout_prob=0.,token_dropout=False)
        model=Student.__new__(Student);nn.Module.__init__(model)
        model.student=EsmModel(config);model.mapping=nn.Linear(32,8);model.classifier=nn.Linear(8,2);model.eval()
        ids=torch.tensor([[0,5,6,7,2,1],[0,8,2,1,1,1]])
        attention=(ids!=1).long();residue=(ids!=1)&(ids!=0)&(ids!=2)
        captured=[];hook=model.student.encoder.layer[-1].register_forward_pre_hook(lambda m,a:captured.append(a[0].detach()))
        with torch.no_grad():
            full=model(ids,attention,residue);hook.remove()
            cached=model(attention_mask=attention,residue_mask=residue,prefix=captured[0])
            for a,b in zip(full,cached):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            alone=model(ids[:1,:5],attention[:1,:5],residue[:1,:5])
            torch.testing.assert_close(full[0][:1],alone[0],rtol=1e-5,atol=1e-6)

    def test_output_kd_has_no_feature_or_prototype_terms(self):
        configure_arm('kd_output')
        self.assertEqual(CONFIG['feature_mse_weight'],0.)
        self.assertEqual(CONFIG['prototype_query_weight'],0.)
        self.assertEqual(CONFIG['prototype_geometry_weight'],0.)
        configure_arm('spmd')

if __name__=='__main__':unittest.main()
