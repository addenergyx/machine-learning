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
        app.config['SECRET_KEY'] = 'secret'
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
        response = self.app.post('/', data=dict(comment="AGTCGCGGATGCGGATGATCGATCGATCGATTAGTTTCGATCGAGGCTAGAT"), follow_redirects=True)
        self.assertIn(b'Back', response.data)
           
    
#client = app.test_client()            
#app = Flask(__name__)    
#with app.test_request_context('/'): 
#    r = client.post(
#        '/',
#        data='sadfeawfdsa',
#        follow_redirects=True
#    )
#    messages = flask.get_flashed_messages()
#    r.data
#        
#    def test_flashy(self):
#        with self.client:
#           app.config['SECRET_KEY'] = 'secret'
#           response = self.client.post(
#                '/',
#                data='sadfeawfdsa',
#                follow_redirects=True
#            )
#           self.assertEqual(response.status_code, 200)
#           self.assertIn(b'Error', response.data)
#        
#        
#        
#client = app.test_client()        
#with client:
#   app.config['SECRET_KEY'] = 'secret'
#   client.post(
#        '/',
#        data='sadfeawfdsa',
#        follow_redirects=True
#    )
#   response = client.get(
#        '/',
#        follow_redirects=False
#    )
#   messages = flask.get_flashed_messages()
#   self.assertEqual(response.status_code, 200)
#   client.assertIn(b'Error', response.data)        
#        
        
#    # Enter invalid sequence
#    def test_flashes(self):
#        app.config['SECRET_KEY'] = 'secret'
#        client = app.test_client()
#        response = self.client.post('/', data='GATTAGTT', follow_redirects=True)
#        messages = flask.get_flashed_messages()
#
#        
#        
#    def test_flash(self):
#        # attempt login with wrong credentials
#        response = self.client.post('/', data='GATTAGTT', follow_redirects=True)
#        self.assertTrue(re.search('Invalid',response.get_data(as_text=True)))
#
#        
#    def test_invalid_data(self):
#        client = app.test_client(self)
#        app.config['SECRET_KEY'] = 'secret'
#
#        #app.secret_key = 'secret'
#
#        #response = client.post('/', data="sdfgdsrgfATCGATCGATTAGTTTCGATfgdgkkCGAGGCTAGAT", content_type='html/text', follow_redirects=True)
#        #assert 'Error' in response.data
#        
#        expected_flash_message = 'Invalid Sequence'
#        
#        response = client.get('/')
#        with client.session_transaction() as session:
#            flash_message = dict(session['_flashes']).get('warning')
#            
#        self.assertIsNotNone(flash_message, session['_flashes'])
#        self.assertEqual(flash_message, expected_flash_message)
#        
#    def test_flash_a_success_message(self):
#        client = app.test_client(self)
#        response = client.post('/', data="sdfgdsrgfATCGATCGATTAGTTTCGATfgdgkkCGAGGCTAGAT",follow_redirects=True)
#        self.assertTrue(b'Invalid' in response.data)
#    
    
    
    
if __name__ == "__main__":
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    