#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import unittest
import re
import flask

# Gets flask application
from app import app 
 
class BasicTests(unittest.TestCase):
     
    #http://flask.pocoo.org/docs/1.0/testing/#the-application
    def setUp(self):
        print("==> Setting up test env")
        app.config['TESTING'] = True
        #app.config['SECRET_KEY'] = 'secret'
        self.app = app.test_client(self) # Creates test instance

    def tearDown(self):
        print("==> Tearing down after tests")
    
    # Ensure that Flask was set up correctly
    def test_index(self):
        response = self.app.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)
    
    # Ensure Home page loads correctly     
    def test_home_page_loads(self):
        response = self.app.get('/', content_type='html/text')
        self.assertIn(b'Simply type or paste your sequence below and click Predict.' ,response.data)
        
    # Ensure cannot view /predict page without entering sequence first    
    def test_predict(self):
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 405)
        
    # Ensure prediction page loads after entering valid sequence
    def test_valid_data(self):
        response = self.app.post('/predict', data=dict(comment="AGTCGCGGATGCGGATGATCGATCGATCGATTAGTTTCGATCGAGGCTAGAT"), follow_redirects=True)
        self.assertIn(b'Back', response.data)
    
    # Ensure Error message appears when enter invalid sequence
    def test_flash(self):
        response = self.app.post('/predict', data=dict(comment="AGTCGCsdfGGATGCGGATGATCGATCGATCGATTAGTTTCGATCGAGGCTAGAT"), follow_redirects=True)
        self.assertIn(b'Error', response.data)

    
if __name__ == "__main__":
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    