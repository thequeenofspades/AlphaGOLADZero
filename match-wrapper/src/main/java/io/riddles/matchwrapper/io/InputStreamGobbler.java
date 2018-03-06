// Copyright 2016 riddles.io (developers@riddles.io)

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//  
//    For the full copyright and license information, please view the LICENSE
//    file that was distributed with this source code.

package io.riddles.matchwrapper.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * InputStreamGobbler class
 *
 * Keeps trying to read output from given process input/error stream. Stores read value and
 * sets a response to read value if it is not the error stream.
 * 
 * @author Jim van Eeden <jim@riddles.io>
 */
public class InputStreamGobbler extends Thread {

    private final long BUFFER_LIMIT = 1000000; // 1 Mb
    
    private InputStream inputStream;
    private IOWrapper wrapper;
    private String type;
    private StringBuffer buffer;
    private boolean finished;

    InputStreamGobbler(InputStream inputStream, IOWrapper wrapper, String type) {
        this.inputStream = inputStream;
        this.wrapper = wrapper;
        this.type = type;
        this.buffer = new StringBuffer();
        this.finished = false;
    }

    /**
     * Read input from stream and store
     */
    public void run() {
        String lastLine;

        try {
            InputStreamReader inputStreamReader = new InputStreamReader(this.inputStream);
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);

            while (!this.finished && (lastLine = bufferedReader.readLine()) != null) {

                // Catches bots that return way too much (infinite loop)
                if (this.buffer.length() >= BUFFER_LIMIT) {
                    String message = String.format(
                            "... %s limit of %s Mb reached. " +
                            "Processing of stream stopped.",
                            this.type,
                            BUFFER_LIMIT / 1000000
                    );

                    this.buffer.append(message);
                    finish();
                    break;
                }

                if (this.type.equals("output")) {
                   this.wrapper.response = lastLine;
                   if (this.wrapper.inputQueue != null) {
                       this.wrapper.inputQueue.add(lastLine);
                   }
                }

                this.buffer.append(lastLine).append("\n");
            }
            try {
                bufferedReader.close();
            } catch (IOException ignored) {}

        } catch (IOException ex) {
            System.err.println(String.format("Readline failed: %s, type: %s", ex, this.type));
        }
    }
    
    /**
     * @return All the data read by this InputStreamGobbler
     */
    public String getData() {
        return buffer.toString();
    }
    
    /**
     * Stop running
     */
    public void finish() {
        this.finished = true;
    }
}
