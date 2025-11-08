import traci
import numpy as np
import random
import timeit
import os
import csv
from encoder_decoder_model import EncoderDecoderPredictor, EncoderDecoderAutoencoder

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs, encoder_decoder_predictor):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = {"SUMO": [], "EncoderDecoder": []}
        self._cumulative_wait_store = {"SUMO": [], "EncoderDecoder": []}
        self._avg_queue_length_store = {"SUMO": [], "EncoderDecoder": []}
        self._training_epochs = training_epochs
        self._sum_queue_length = 0  # Initialize _sum_queue_length
        self._sum_waiting_time = 0  # Initialize _sum_waiting_time
        self.encoder_decoder_predictor = encoder_decoder_predictor
        self.state_buffer = []  # Buffer to store last sequence_length SUMO states
        self.prediction_interval = 10
        self.min_buffer_size = 9
        self.sequence_memory = []  # Store (sequence, actual_next_state) pairs
        self.ed_predictions = []  # Store (prediction, actual) pairs for evaluation
        self.anomaly_detector = EncoderDecoderAutoencoder(num_states, self.min_buffer_size) if encoder_decoder_predictor else None
        self._waiting_times = {}
        
    def train_encoder_decoder_with_batch(self):
        """Train Encoder-Decoder with accumulated data in batches"""
        if not self.encoder_decoder_predictor or len(self.sequence_memory) < 32:  # Wait for enough samples
            return
            
        # Sample batch from memory
        batch_indices = random.sample(range(len(self.sequence_memory)), min(32, len(self.sequence_memory)))
        X = np.array([self.sequence_memory[i][0] for i in batch_indices])
        y = np.array([self.sequence_memory[i][1] for i in batch_indices])
        
        # Train model
        self.encoder_decoder_predictor.model.fit(X, y, epochs=3, batch_size=min(32, len(batch_indices)), verbose=0)

    def run(self, episode, epsilon, model_type):
        """
        Runs an episode of simulation, then starts a training session.
        """
        start_time = timeit.default_timer()

        # Generate route file and start SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print(f"Simulating episode {episode} with {model_type} model...")

        # Create CSV files to store episode data
        original_csv_path = os.path.join('csv', f"states_episode_{episode}.csv")
        ed_csv_path = os.path.join('ed_csv', f"encoder_decoder_episode_{episode}.csv")
        
        # Ensure directories exist
        os.makedirs('csv', exist_ok=True)
        os.makedirs('ed_csv', exist_ok=True)
        
        with open(original_csv_path, 'w') as orig_file, open(ed_csv_path, 'w') as ed_file:
            orig_writer = csv.writer(orig_file)
            ed_writer = csv.writer(ed_file)
            
            # Write headers
            orig_writer.writerow(["timestamp", "states", "source"])
            ed_writer.writerow(["timestamp", "states", "source"])
            
            # Initialize simulation with yellow and green phases
            self._simulate(self._yellow_duration, orig_writer, ed_writer)
            self._simulate(self._green_duration, orig_writer, ed_writer)
            
            # Initialize variables
            self._step = 0
            self._waiting_times = {}
            self._sum_neg_reward = 0
            old_total_wait = 0
            old_state = -1
            old_action = -1
            
            # Clear state buffer at the beginning of episode
            self.state_buffer = []
            
            while self._step < self._max_steps:
                # Get current state of the intersection
                sumo_state = self._get_state()
                current_state = sumo_state  # Default to SUMO state
                source = "SUMO"  # Default source
                
                # Debug output for model type
                if self._step == 0:
                    print(f"Running simulation with model_type: {model_type}")
                    print(f"Encoder-decoder predictor available: {self.encoder_decoder_predictor is not None}")
                
                # Add state to buffer
                self.state_buffer.append(sumo_state)
                if len(self.state_buffer) > self.min_buffer_size:
                    self.state_buffer.pop(0)
                    
                # Model selection logic - determine which state to use
                if model_type == "EncoderDecoder" and len(self.state_buffer) >= self.min_buffer_size:
                    if self._step % self.prediction_interval == 0:  # Predict every 10th step
                        # Debug output
                        print(f"Step {self._step}: Using EncoderDecoder prediction (buffer size: {len(self.state_buffer)})")
                        
                        # Use Encoder-Decoder prediction
                        ed_state = self.encoder_decoder_predictor.predict(np.array(self.state_buffer))
                        current_state = ed_state
                        source = "EncoderDecoder"
                        
                        # Store this prediction and the future actual state for evaluation
                        self.ed_predictions.append((ed_state, None))  # Will fill actual state later
                    
                    # If we're at step+1 after a prediction, get actual outcome and store in memory
                    elif self._step % self.prediction_interval == 1 and self._step > 1:
                        # Get the actual state that occurred after our last prediction
                        actual_state = sumo_state
                        
                        # Update the last prediction's actual value
                        if self.ed_predictions:
                            self.ed_predictions[-1] = (self.ed_predictions[-1][0], actual_state)
                        
                        # Store in sequence memory for training
                        if len(self.state_buffer) >= self.min_buffer_size:
                            sequence = np.array(self.state_buffer[-(self.min_buffer_size):])
                            self.sequence_memory.append((sequence, actual_state))
                            
                            # Limit memory size
                            if len(self.sequence_memory) > 1000:
                                self.sequence_memory.pop(0)
                        
                        # Every Nth prediction, do batch training
                        if len(self.ed_predictions) % 5 == 0:  # Every 5 predictions
                            self.train_encoder_decoder_with_batch()
                            
                # Anomaly detection
                if self.anomaly_detector and len(self.state_buffer) >= self.min_buffer_size:
                    try:
                        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly([self.state_buffer])
                        if is_anomaly:
                            print(f"Anomaly detected at step {self._step}, score: {anomaly_score:.4f}")
                    except:
                        pass  # Skip if anomaly detector not trained yet
                        
                # Write to CSVs
                self._write_to_csv(self._step, sumo_state, current_state, source, orig_writer, ed_writer)
                
                # Calculate reward of previous action
                current_total_wait = self._collect_waiting_times()
                reward = old_total_wait - current_total_wait
                
                # Track negative rewards for plotting
                if reward < 0:
                    self._sum_neg_reward += reward
                    
                # Save data into memory for RL training
                if self._step != 0:
                    self._Memory.add_sample((old_state, old_action, reward, current_state))
                
                # Choose action based on current state
                action = self._choose_action(current_state, epsilon)
                
                # Activate yellow phase if action changes
                if self._step != 0 and old_action != action:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._yellow_duration, orig_writer, ed_writer)
                
                # Activate green phase and simulate
                self._set_green_phase(action)
                self._simulate(self._green_duration, orig_writer, ed_writer)
                
                # Update variables for next iteration
                old_state = current_state
                old_action = action
                old_total_wait = current_total_wait

                print(f"Step {self._step} completed...")
        
        # Store metrics for this episode in the appropriate dictionary key
        self._reward_store[model_type].append(self._sum_neg_reward)
        self._cumulative_wait_store[model_type].append(self._sum_waiting_time)
        self._avg_queue_length_store[model_type].append(self._sum_queue_length / self._max_steps)
        
        print(f"Model: {model_type}, Total negative reward: {self._sum_neg_reward}, Epsilon: {round(epsilon, 2)}")
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        
        print("Training RL model...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)
        
        return simulation_time, training_time
    def _simulate(self, steps_todo, orig_writer, ed_writer):
        """
        Execute steps in sumo while gathering statistics and recording states at each step.
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
        
            # Record state at every step
            current_state = self._get_state()
            
            # Gather statistics
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_length == waited_seconds

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN

    def _write_to_csv(self, step, sumo_state, ed_state, source, orig_writer, ed_writer):
        """
        Writes state information to the respective CSV files.
        - orig_writer logs all SUMO states.
        - ed_writer logs 9 SUMO states + 1 Encoder-Decoder-predicted state every 10th step.
        """
        # Write all SUMO states to original CSV
        orig_writer.writerow([step, list(sumo_state), "SUMO"])

        # Write to Encoder-Decoder CSV based on the source (SUMO or EncoderDecoder)
        if source == "SUMO":
            ed_writer.writerow([step, list(sumo_state), "SUMO"])
        elif source == "EncoderDecoder":
            ed_writer.writerow([step, list(ed_state), "EncoderDecoder"])

    def pre_train_encoder_decoder(self, episodes=5):
        """Pre-train Encoder-Decoder model on collected sequences from initial episodes"""
        print("Starting pre-training...")
        if not self.encoder_decoder_predictor:
            return  # Only run if we have a predictor
            
        print("Pre-training Encoder-Decoder model...")
        collected_sequences = []
        collected_targets = []
        
        # Run simulation to collect training data
        for episode in range(episodes):
            print(f"Pre-training episode {episode}...")
            self._TrafficGen.generate_routefile(seed=episode)
            traci.start(self._sumo_cmd)
            states = []
            
            # Collect states
            for step in range(self._max_steps):
                traci.simulationStep()
                state = self._get_state()
                states.append(state)
                
            # Create training sequences
            for i in range(len(states) - self.min_buffer_size):
                sequence = states[i:i+self.min_buffer_size]
                target = states[i+self.min_buffer_size]
                collected_sequences.append(sequence)
                collected_targets.append(target)
            
            traci.close()
        
        # Convert to numpy arrays and train
        X = np.array(collected_sequences)
        y = np.array(collected_targets)
        self.encoder_decoder_predictor.model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        
        # Train anomaly detector with normal patterns
        if self.anomaly_detector:
            print("Training anomaly detector...")
            self.anomaly_detector.train_on_normal_traffic(X)
        
        print("Pre-training complete")

    def evaluate_encoder_decoder_accuracy(self):
        """Calculate and return Encoder-Decoder prediction accuracy metrics"""
        if not hasattr(self, 'ed_predictions') or not self.ed_predictions:
            return {}
        
        # Filter out entries with None actual values
        valid_predictions = [(pred, actual) for pred, actual in self.ed_predictions if actual is not None]
        
        if not valid_predictions:
            return {}
            
        mse = np.mean([np.mean((pred - actual)**2) for pred, actual in valid_predictions])
        mae = np.mean([np.mean(np.abs(pred - actual)) for pred, actual in valid_predictions])
        
        # Calculate relative error (in percentage)
        relative_errors = []
        for pred, actual in valid_predictions:
            # Avoid division by very small numbers
            norm_actual = np.linalg.norm(actual)
            if norm_actual > 1e-10:
                rel_err = np.linalg.norm(pred - actual) / norm_actual * 100
                relative_errors.append(rel_err)
        
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0
                    
        return {'MSE': mse, 'MAE': mae, 'RelError': avg_rel_error}

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

